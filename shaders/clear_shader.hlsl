RWTexture2D<float4> Target;
RWBuffer<uint> Stats;
                      
[numthreads(8,8,1)]
void main(uint3 tid: SV_DispatchThreadId)
{
    Target[tid.xy] = float4(1,0,0,1);
    Stats[0] = 0;
}