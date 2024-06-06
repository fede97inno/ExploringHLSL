RWTexture2D<float4> Target;
RWTexture2D<uint> Depth;
RWBuffer<uint> Stats;
                      
[numthreads(8,8,1)]
void main(uint3 tid: SV_DispatchThreadId)
{
    Target[tid.xy] = float4(1,0,0,1);
    Stats[0] = 0;
    Depth[tid.xy] = 0xFFFFFFFF;
}