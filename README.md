# NvColorSpace
Converts the image's color using NVIDIA CUDA

## Supported
- RGBA to BGRA
- RGBA to BGR
- BGRA to YUV

[Want other method?](https://github.com/MineEric64/NvColorSpace/issues/new)

## [Latest Release](https://github.com/MineEric64/NvColorSpace/releases)

## C# Wrapper
If you want to use these methods in C#, you have to download the NvColorSpace DLL and make wrapper like:
```c#
public class NvColorSpace
{
    public const string DLL_NAME = "NvColorSpace.dll";

    // ---

    [DllImport(DLL_NAME, EntryPoint = "RGBA32ToBGRA32")]
    public static extern int RGBA32ToBGRA32(IntPtr rgba, IntPtr bgra, int width, int height);

    [DllImport(DLL_NAME, EntryPoint = "RGBA32ToBGR24")]
    public static extern int RGBA32ToBGR24(IntPtr rgba, IntPtr bgr, int width, int height);
}
```

and use the method like:
```c#
int width = 1920;
int height = 1080;

byte[] rgbaBuffer = ...; //get the buffer from source you want
byte[] bgrBuffer = new byte[width * height * 3];

IntPtr rgbaPtr = Marshal.AllocHGlobal(rgbaBuffer.Length);
IntPtr bgrPtr = Marshal.AllocHGlobal(bgrBuffer.Length);

Marshal.Copy(rgbaBuffer, 0, rgbaPtr, rgbaBuffer.Length); //copy the buffer to pointer

int status = NvColorSpace.RGBA32ToBGR24(rgbaPtr, bgrPtr, width, height); //int type can be changed to cudaError_t

if (status == 0) {
    Marshal.Copy(bgrPtr, bgrBuffer, 0, bgrBuffer.Length); //copy the pointer's data to buffer
    
    //now you can use bgrBuffer!
    
}
else {
    Console.WriteLine("Error occured! More informations at the file (BetterNvLog.log) or googling status code. ex) cuda runtime error 9");
}

Marshal.FreeHGlobal(rgbaPtr);
Marshal.FreeHGlobal(bgrPtr);
```
