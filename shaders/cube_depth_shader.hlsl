RWTexture2D<float4> Target;
Texture2D<uint> Depth;
Buffer<float3> VertexBuffer;
Buffer<uint3> IndexBuffer;
Buffer<float3> ColorBuffer;

struct Config {

    float4x4 PerspectiveMatrix; // -> 4 * 4 * 4 
    float4x4 CameraMatrix;      // -> 4 * 4 * 4 
    float4x4 WorldMatrix;
    float DeltaX;               // 4
    float DeltaY;               // 4
    uint TriangleNum;           // 4
};
                      
ConstantBuffer<Config> ConfigBuffer;
RWBuffer<uint> Stats;
                      
float2 NDCtoPixel(float2 Position, uint width, uint height)
{
    const float x = (Position.x + 1) * 0.5 * width;                      
    const float y = (1 - Position.y) * 0.5 * height;      
    return float2(x, y);
}              

float3 barycentric(float2 a, float2 b, float2 c, float2 p)
{
    float3 x = float3(c.x - a.x, b.x - a.x, a.x - p.x);
    float3 y = float3(c.y - a.y, b.y - a.y, a.y - p.y);
    float3 u = cross(x, y);

    if (abs(u.z) < 1.0)
    {
        return float3(-1, 1, 1);
    }

    return float3(1.0 - (u.x+u.y)/u.z, u.y/u.z, u.x/u.z);
}        
                      
[numthreads(8,8,8)]
void main(uint3 tid: SV_DispatchThreadId)
{      
    uint width;
    uint height;
    Target.GetDimensions(width, height);    

    const uint index = tid.z;

    uint index0 = IndexBuffer[index].x;
    uint index1 = IndexBuffer[index].y;
    uint index2 = IndexBuffer[index].z;

    // Original Point
    float4 trianglePoint0 = mul(float4(VertexBuffer[index0], 1), ConfigBuffer.WorldMatrix);
    float4 trianglePoint1 = mul(float4(VertexBuffer[index1], 1), ConfigBuffer.WorldMatrix);
    float4 trianglePoint2 = mul(float4(VertexBuffer[index2], 1), ConfigBuffer.WorldMatrix);

    trianglePoint0 = mul(trianglePoint0, ConfigBuffer.CameraMatrix);
    trianglePoint1 = mul(trianglePoint1, ConfigBuffer.CameraMatrix);
    trianglePoint2 = mul(trianglePoint2, ConfigBuffer.CameraMatrix);

    // Perspective correction
    float4 point0 = mul(trianglePoint0, ConfigBuffer.PerspectiveMatrix);
    float4 point1 = mul(trianglePoint1, ConfigBuffer.PerspectiveMatrix);
    float4 point2 = mul(trianglePoint2, ConfigBuffer.PerspectiveMatrix);

    point0.xyz /= point0.w;
    point1.xyz /= point1.w;
    point2.xyz /= point2.w;

    float2 pointPixel0 = NDCtoPixel(point0.xy, width, height);
    float2 pointPixel1 = NDCtoPixel(point1.xy, width, height);
    float2 pointPixel2 = NDCtoPixel(point2.xy, width, height);
    
    float3 BC = barycentric(pointPixel0, pointPixel1, pointPixel2, tid.xy);
                    
    if(BC.x < 0|| BC.y < 0 || BC.z < 0)
    {
        return;
    }
    else
    {
        const float z = point0.z * BC.x + point1.z * BC.y + point2.z * BC.z;

        if(z < 0)
        {
            return;
        }

        if(z <= asfloat(Depth[tid.xy]))
        {
            Target[tid.xy] = float4(ColorBuffer[index], 1);
            InterlockedAdd(Stats[0], 1);        
        }
    }
}
