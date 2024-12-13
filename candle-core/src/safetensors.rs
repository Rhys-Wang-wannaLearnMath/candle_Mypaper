use crate::{DType, Device, Error, Result, Tensor, WithDType}; // 引入本地模块中的类型
use safetensors::tensor as st; // 将 `safetensors::tensor` 模块重命名为 `st` 以便于引用
use safetensors::tensor::SafeTensors; // 引入 `SafeTensors` 序列化类型
use std::borrow::Cow; // `Cow` 用于表示借用或拥有的值
use std::collections::HashMap; // 引入哈希映射
use std::path::Path; // 用于处理文件路径

/// 将 `DType`（内部定义的数据类型）转换为 `safetensors` 中的 `Dtype`
impl From<DType> for st::Dtype {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => st::Dtype::U8,
            DType::U32 => st::Dtype::U32,
            DType::I64 => st::Dtype::I64,
            DType::BF16 => st::Dtype::BF16,
            DType::F16 => st::Dtype::F16,
            DType::F32 => st::Dtype::F32,
            DType::F64 => st::Dtype::F64,
        }
    }
}

/// 尝试从 `safetensors::Dtype` 转换为内部的 `DType`
/// 如果 `safetensors::Dtype` 不支持则返回错误
impl TryFrom<st::Dtype> for DType {
    type Error = Error;
    fn try_from(value: st::Dtype) -> Result<Self> {
        match value {
            st::Dtype::U8 => Ok(DType::U8),
            st::Dtype::U32 => Ok(DType::U32),
            st::Dtype::I64 => Ok(DType::I64),
            st::Dtype::BF16 => Ok(DType::BF16),
            st::Dtype::F16 => Ok(DType::F16),
            st::Dtype::F32 => Ok(DType::F32),
            st::Dtype::F64 => Ok(DType::F64),
            dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)), // 不支持的类型报错
        }
    }
}

/// 实现 `safetensors::View` trait，用于获取张量的类型、形状和数据
impl st::View for Tensor {
    fn dtype(&self) -> st::Dtype {
        self.dtype().into() // 调用 `From<DType> for st::Dtype` 进行转换
    }
    fn shape(&self) -> &[usize] {
        self.shape().dims() // 获取张量的形状
    }

    fn data(&self) -> Cow<[u8]> {
        // 从 GPU 拷贝到 CPU，并返回数据
        // TODO: 这里应该避免 unwrap
        Cow::Owned(convert_back(self).unwrap())
    }

    fn data_len(&self) -> usize {
        let n: usize = self.shape().elem_count(); // 计算总元素数量
        let bytes_per_element = self.dtype().size_in_bytes(); // 获取每个元素的字节数
        n * bytes_per_element // 总字节数
    }
}

/// 对 `&Tensor` 实现 `safetensors::View` trait
/// 与之前实现不同的是，这里操作的是张量的引用
impl st::View for &Tensor {
    fn dtype(&self) -> st::Dtype {
        (*self).dtype().into() // 解引用并调用 `dtype()`
    }
    fn shape(&self) -> &[usize] {
        self.dims() // 获取张量的维度
    }

    fn data(&self) -> Cow<[u8]> {
        // 从 GPU 拷贝到 CPU，并返回数据
        // TODO: 这里应该避免 unwrap
        Cow::Owned(convert_back(self).unwrap())
    }

    fn data_len(&self) -> usize {
        let n: usize = self.dims().iter().product(); // 计算总元素数量
        let bytes_per_element = (*self).dtype().size_in_bytes(); // 获取每个元素的字节数
        n * bytes_per_element // 总字节数
    }
}

/// 张量的保存功能：保存为 `safetensors` 格式
impl Tensor {
    pub fn save_safetensors<P: AsRef<Path>>(&self, name: &str, filename: P) -> Result<()> {
        let data = [(name, self.clone())]; // 创建包含张量的数组
        Ok(st::serialize_to_file(data, &None, filename.as_ref())?) // 序列化为文件
    }
}

/// 将字节切片转换为张量
fn convert_slice<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes(); // 每个元素的字节大小
    let elem_count = data.len() / size_in_bytes; // 元素总数
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // 安全：检查数据对齐，若对齐则安全转换
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device) // 从切片创建张量
    } else {
        // 若不对齐，则将数据拷贝到新的向量
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: 确保内存是连续的，并且不会与当前数据重叠
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device) // 从切片创建张量
    }
}

/// 带转换功能的切片转换函数，支持类型转换
fn convert_slice_with_cast<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    data: &[u8],
    shape: &[usize],
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    let size_in_bytes = std::mem::size_of::<T>(); // 计算 T 类型的字节大小
    let elem_count = data.len() / size_in_bytes; // 元素总数
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY: 检查数据对齐
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        let data = data.iter().map(|t| conv(*t)).collect::<Result<Vec<_>>>()?; // 执行类型转换
        Tensor::from_vec(data, shape, device) // 转换为张量
    } else {
        // 若不对齐则需要拷贝数据
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: 确保内存是连续的
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        let c = c.into_iter().map(conv).collect::<Result<Vec<_>>>()?; // 执行转换
        Tensor::from_vec(c, shape, device) // 转换为张量
    }
}

/// 从 `SafeTensors` 的视图中加载张量并执行类型转换
fn convert_with_cast_<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    view: &st::TensorView<'_>,
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    convert_slice_with_cast::<T, U, F>(view.data(), view.shape(), device, conv) // 使用转换函数
}

/// 从 `SafeTensors` 的视图中加载张量，不进行类型转换
fn convert_<T: WithDType>(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    convert_slice::<T>(view.data(), view.shape(), device) // 直接加载为目标类型
}

/// 将张量转换为字节切片，方便序列化或存储
fn convert_back_<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let size_in_bytes = T::DTYPE.size_in_bytes(); // 获取每个元素的字节数
    let length = vs.len() * size_in_bytes; // 计算总字节数
    let capacity = vs.capacity() * size_in_bytes; // 计算总容量
    let ptr = vs.as_mut_ptr() as *mut u8; // 将 `Vec<T>` 转换为 `Vec<u8>`
    std::mem::forget(vs); // 避免调用析构函数
    unsafe { Vec::from_raw_parts(ptr, length, capacity) } // 返回转换后的字节数组
}

/// Trait 用于加载张量
pub trait Load {
    fn load(&self, device: &Device) -> Result<Tensor>; // 从 `SafeTensors` 视图加载张量
}

/// 对 `safetensors::TensorView` 实现加载功能
impl<'a> Load for st::TensorView<'a> {
    fn load(&self, device: &Device) -> Result<Tensor> {
        convert(self, device) // 调用 `convert` 加载张量
    }
}

/// 张量构造函数，从原始缓冲区中创建张量
impl Tensor {
    pub fn from_raw_buffer(
        data: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &Device,
    ) -> Result<Self> {
        match dtype {
            DType::U8 => convert_slice::<u8>(data, shape, device), // 根据数据类型选择合适的转换函数
            DType::U32 => convert_slice::<u32>(data, shape, device),
            DType::I64 => convert_slice::<i64>(data, shape, device),
            DType::BF16 => convert_slice::<half::bf16>(data, shape, device),
            DType::F16 => convert_slice::<half::f16>(data, shape, device),
            DType::F32 => convert_slice::<f32>(data, shape, device),
            DType::F64 => convert_slice::<f64>(data, shape, device),
        }
    }
}

/// 根据 `safetensors` 的数据类型加载张量
fn convert(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    match view.dtype() {
        st::Dtype::U8 => convert_::<u8>(view, device), // 无需转换，直接加载
        st::Dtype::U16 => {
            let conv = |x| Ok(u32::from(x)); // 将 u16 转换为 u32
            convert_with_cast_::<u16, u32, _>(view, device, conv)
        }
        st::Dtype::U32 => convert_::<u32>(view, device),
        st::Dtype::I32 => {
            let conv = |x| Ok(i64::from(x)); // 将 i32 转换为 i64
            convert_with_cast_::<i32, i64, _>(view, device, conv)
        }
        st::Dtype::I64 => convert_::<i64>(view, device),
        st::Dtype::BF16 => convert_::<half::bf16>(view, device),
        st::Dtype::F16 => convert_::<half::f16>(view, device),
        st::Dtype::F32 => convert_::<f32>(view, device),
        st::Dtype::F64 => convert_::<f64>(view, device),
        dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)), // 不支持的数据类型返回错误
    }
}

/// 将张量转换回字节数组
fn convert_back(tensor: &Tensor) -> Result<Vec<u8>> {
    let tensor = tensor.flatten_all()?; // 将张量展平
    match tensor.dtype() {
        DType::U8 => Ok(convert_back_::<u8>(tensor.to_vec1()?)), // 根据数据类型选择转换函数
        DType::U32 => Ok(convert_back_::<u32>(tensor.to_vec1()?)),
        DType::I64 => Ok(convert_back_::<i64>(tensor.to_vec1()?)),
        DType::F16 => Ok(convert_back_::<half::f16>(tensor.to_vec1()?)),
        DType::BF16 => Ok(convert_back_::<half::bf16>(tensor.to_vec1()?)),
        DType::F32 => Ok(convert_back_::<f32>(tensor.to_vec1()?)),
        DType::F64 => Ok(convert_back_::<f64>(tensor.to_vec1()?)),
    }
}

/// 从文件加载 `SafeTensors`，并将其转换为张量
pub fn load<P: AsRef<Path>>(filename: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(filename.as_ref())?; // 读取文件
    load_buffer(&data[..], device) // 从缓冲区加载
}

/// 从字节缓冲区加载 `SafeTensors`
pub fn load_buffer(data: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let st = safetensors::SafeTensors::deserialize(data)?; // 反序列化
    st.tensors()
        .into_iter()
        .map(|(name, view)| Ok((name, view.load(device)?))) // 加载每个张量
        .collect()
}

/// 将张量保存为 `SafeTensors` 格式文件
pub fn save<K: AsRef<str> + Ord + std::fmt::Display, P: AsRef<Path>>(
    tensors: &HashMap<K, Tensor>,
    filename: P,
) -> Result<()> {
    Ok(st::serialize_to_file(tensors, &None, filename.as_ref())?) // 序列化张量并保存到文件
}

// 该宏实现将结构体 `SafeTensors_` 声明为可使用 `yoke` 的 `Yokeable` trait。
// `SafeTensors_` 是对 `SafeTensors` 的一个简单封装，用于在内存映射中安全使用。
#[derive(yoke::Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>); // 'a 是生命周期参数，指向 SafeTensors 的数据引用

// `MmapedSafetensors` 用于管理多个通过 `vec<u8>.` 加载的 `SafeTensors` 文件。
pub struct MmapedSafetensors {
    // 一个 Yoke 类型的 Vec，保存了通过内存映射加载的 SafeTensors。
    safetensors: Vec<yoke::Yoke<SafeTensors_<'static>, Vec<u8>>>,

    // 路由表，用于根据张量的名称快速查找它所在的文件（通过文件的索引）。
    routing: Option<HashMap<String, usize>>,
}

impl MmapedSafetensors {
    /// 创建一个 `MmapedSafetensors` 实例，它包装了通过 `mmap` 加载的文件，并反序列化 `SafeTensors` 头部。
    ///
    /// # Safety
    ///
    /// 使用内存映射是一个不安全操作，涉及直接与文件的内存交互，因此需要通过 `unsafe` 块确保安全。
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref(); // 获取路径

        // 使用内存映射将文件内容映射到进程的地址空间
        let file = std::fs::read(p).map_err(|e| Error::from(e).with_path(p))?;

        // 使用 `Yoke` 将 `SafeTensors_` 类型绑定到内存映射的数据
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, Vec<u8>>::try_attach_to_cart(
            file,
            |data: &[u8]| {
                // 反序列化 `SafeTensors` 数据
                let st = safetensors::SafeTensors::deserialize(data)
                    .map_err(|e| Error::from(e).with_path(p))?;
                Ok::<_, Error>(SafeTensors_(st)) // 封装成 SafeTensors_
            },
        )?;

        // 创建 MmapedSafetensors 实例，初始状态下 routing 表为空
        Ok(Self {
            safetensors: vec![safetensors], // 将反序列化的 `SafeTensors` 添加到列表中
            routing: None,                  // 初始时还没有路由表
        })
    }

    /// 创建一个包装多个内存映射文件的实例，并反序列化每个文件的 `SafeTensors` 头部。
    ///
    /// 如果某个张量在多个文件中存在，最后的文件将会覆盖前面的同名张量。
    ///
    /// # Safety
    ///
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut routing = HashMap::new(); // 初始化路由表
        let mut safetensors: Vec<yoke::Yoke<SafeTensors_<'static>, Vec<u8>>> = vec![]; // 保存加载的文件

        // 遍历每个路径，逐个加载文件
        for (index, p) in paths.iter().enumerate() {
            let p = p.as_ref();
            // 直接读取文件内容到 Vec<u8>
            let file_data = std::fs::read(p).map_err(|e| Error::from(e).with_path(p))?;

            // 使用 Yoke 将 SafeTensors 数据绑定到 Vec<u8>
            let data = yoke::Yoke::<SafeTensors_<'static>, Vec<u8>>::try_attach_to_cart(
                file_data,
                |data: &[u8]| {
                    // 反序列化 SafeTensors
                    let st = safetensors::SafeTensors::deserialize(data)
                        .map_err(|e| Error::from(e).with_path(p))?;
                    Ok::<_, Error>(SafeTensors_(st)) // 封装成 SafeTensors_
                },
            )?;

            // 将文件中的张量名称映射到文件索引，以便后续查找
            for k in data.get().0.names() {
                routing.insert(k.to_string(), index);
            }
            safetensors.push(data); // 添加到 safetensors 列表中
        }

        Ok(Self {
            safetensors,            // 保存所有加载的 SafeTensors
            routing: Some(routing), // 保存路由表
        })
    }

    /// 根据名称加载张量。
    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.load(dev) // 通过名字获取张量，并从设备加载
    }

    /// 返回所有张量的 (名称, TensorView) 列表。
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![]; // 初始化空列表
                                  // 从每个 safetensors 文件中获取张量视图
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.get().0.tensors())
        }
        tensors.into_iter().flatten().collect() // 展平成一个 Vec
    }

    /// 根据张量名称获取张量视图。
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        let index = match &self.routing {
            None => 0, // 如果没有路由表，默认使用第一个文件
            Some(routing) => {
                let index = routing.get(name).ok_or_else(|| {
                    // 如果找不到张量名，返回错误
                    Error::CannotFindTensor {
                        path: name.to_string(),
                    }
                    .bt()
                })?;
                *index // 找到对应的文件索引
            }
        };
        // 返回指定文件中的张量视图
        Ok(self.safetensors[index].get().0.tensor(name)?)
    }
}

// 使用内存切片加载 `SafeTensors` 的封装类型
pub struct SliceSafetensors<'a> {
    safetensors: SafeTensors<'a>, // 内部保存 SafeTensors 数据
}

impl<'a> SliceSafetensors<'a> {
    /// 通过字节缓冲区创建 `SliceSafetensors`，并反序列化 SafeTensors 头部。
    pub fn new(buffer: &'a [u8]) -> Result<Self> {
        let safetensors = safetensors::SafeTensors::deserialize(buffer)?; // 反序列化
        Ok(Self { safetensors }) // 返回实例
    }

    /// 根据张量名称加载张量。
    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.safetensors.tensor(name)?.load(dev) // 从设备加载张量
    }

    /// 返回所有张量的 (名称, TensorView) 列表。
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.tensors() // 返回所有张量视图
    }

    /// 根据名称获取张量视图。
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        Ok(self.safetensors.tensor(name)?) // 获取特定张量的视图
    }
}

// 通过缓冲区加载 `SafeTensors` 的封装类型
pub struct BufferedSafetensors {
    safetensors: yoke::Yoke<SafeTensors_<'static>, Vec<u8>>, // 使用 `Yoke` 包装缓冲区数据
}

impl BufferedSafetensors {
    /// 通过字节缓冲区创建 `BufferedSafetensors`，并反序列化 SafeTensors 头部。
    pub fn new(buffer: Vec<u8>) -> Result<Self> {
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, Vec<u8>>::try_attach_to_cart(
            buffer,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)?; // 反序列化数据
                Ok::<_, Error>(SafeTensors_(st)) // 返回 SafeTensors 封装
            },
        )?;
        Ok(Self { safetensors }) // 返回实例
    }

    /// 根据张量名称加载张量。
    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        self.get(name)?.load(dev) // 获取张量并从设备加载
    }

    /// 返回所有张量的 (名称, TensorView) 列表。
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.get().0.tensors() // 返回张量视图
    }

    /// 根据名称获取张量视图。
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        Ok(self.safetensors.get().0.tensor(name)?) // 获取张量视图
    }
}

// 内存映射文件的封装，允许用户通过内存映射访问张量
pub struct MmapedFile {
    path: std::path::PathBuf, // 文件路径
    inner: memmap2::Mmap,     // 内存映射对象
}

impl MmapedFile {
    /// 创建内存映射文件，并通过 `MmapOptions` 将文件映射到内存中。
    ///
    /// # Safety
    ///
    /// 内存映射是一个不安全操作，涉及直接与底层文件的内存进行交互。
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref(); // 获取路径
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?; // 打开文件
        let inner = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::from(e).with_path(p))?; // 映射文件到内存
        Ok(Self {
            inner,                 // 保存映射后的内存
            path: p.to_path_buf(), // 保存文件路径
        })
    }

    /// 反序列化内存映射文件为 `SafeTensors`。
    pub fn deserialize(&self) -> Result<SafeTensors<'_>> {
        let st = safetensors::SafeTensors::deserialize(&self.inner)
            .map_err(|e| Error::from(e).with_path(&self.path))?; // 反序列化 SafeTensors
        Ok(st) // 返回 SafeTensors 实例
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn save_single_tensor() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        t.save_safetensors("t", "t.safetensors").unwrap();
        let bytes = std::fs::read("t.safetensors").unwrap();
        assert_eq!(bytes, b"@\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
        std::fs::remove_file("t.safetensors").unwrap();
    }

    #[test]
    fn save_load_multiple_tensors() {
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let map: HashMap<_, _> = [("t", t), ("u", u)].into_iter().collect();
        save(&map, "multi.safetensors").unwrap();

        let weights = load("multi.safetensors", &Device::Cpu).unwrap();
        assert_eq!(weights.get("t").unwrap().dims(), &[2, 2]);
        assert_eq!(weights.get("u").unwrap().dims(), &[1, 2]);
        let bytes = std::fs::read("multi.safetensors").unwrap();
        assert_eq!(bytes, b"x\0\0\0\0\0\0\0{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"u\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}      \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
        std::fs::remove_file("multi.safetensors").unwrap();
    }
}
