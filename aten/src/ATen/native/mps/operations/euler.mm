
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/cons2prim_native.h>
#include <cmath>
#include <limits>

namespace at::native {

// scope the MPS's internal methods to not expose them to at::native
namespace mps {

static void cons2prim_mps(const Tensor& cons, const Tensor& prim, double gammad) 
{
  auto stream = getCurrentMPSStream();

  struct CachedGraph: public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* rho = nil;
    MPSGraphTensor* m1 = nil;
    MPSGraphTensor* m2 = nil;
    MPSGraphTensor* m3 = nil;
    MPSGraphTensor* etol = nil;

    MPSGraphTensor* prim = nil;
    MPSGraphTensor* gm1 = nil;
  };

  @autoreleasepool {
    // Create MPSGraph
    string key = "euler_cons2prim_mps_out" + getTensorsStringKey({cons, prim, gammad});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, 
      [&](auto mpsGraph, auto newCachedGraph) {
        newCachedGraph->rho = mpsGraphRankedPlaceHolder(mpsGraph, cons.select(0, 0));
        newCachedGraph->m1 = mpsGraphRankedPlaceHolder(mpsGraph, cons.select(0, 1));
        newCachedGraph->m2 = mpsGraphRankedPlaceHolder(mpsGraph, cons.select(0, 2));
        newCachedGraph->m3 = mpsGraphRankedPlaceHolder(mpsGraph, cons.select(0, 3));
        newCachedGraph->etol = mpsGraphRankedPlaceHolder(mpsGraph, cons.select(0, 4));
        newCachedGraph->gm1 = mpsGraphScalarPlaceHolder(mpsGraph, MPSDataTypeFloat32);

        newCachedGraph->prim = mpsGraphRankedPlaceHolder(mpsGraph, prim);

        // Function logic
        // velocity
        auto v1 = [mpsGraph sliceWithTensor:newCachedGraph->prim
                    dimension:@(0) start:@(1) length:@(1) name:@"v1"];
        auto v2 = [mpsGraph sliceWithTensor:newCachedGraph->prim
                    dimension:@(0) start:@(2) length:@(1) name:@"v2"];
        auto v3 = [mpsGraph sliceWithTensor:newCachedGraph->prim
                    dimension:@(0) start:@(3) length:@(1) name:@"v3"];
        auto pres = [mpsGraph sliceWithTensor:newCachedGraph->prim
                      dimension:@(0) start:@(4) length:@(1) name:@"pres"];

        v1 = [mpsGraph
          divisionWithPrimaryTensor:newCachedGraph->m1
                    secondaryTensor:newCachedGraph->rho name:@"v1"];
        v2 = [mpsGraph 
          divisionWithPrimaryTensor:newCachedGraph->m2
                    secondaryTensor:newCachedGraph->rho name:@"v2"];
        v3 = [mpsGraph
          divisionWithPrimaryTensor:newCachedGraph->m3
                    secondaryTensor:newCachedGraph->rho name:@"v3"];

        // kinetic energy
        auto half = [mpsGraph tensorWithScalar:@(0.5) MPSDataTypeFloat32];
        auto ke1 = [mpsGraph 
          multiplicationWithPrimaryTensor:newCachedGraph->v1
                          secondaryTensor:newCachedGraph->v1 name:nil];
        auto ke2 = [mpsGraph
          multiplicationWithPrimaryTensor:newCachedGraph->v2
                          secondaryTensor:newCachedGraph->v2 name:nil];
        auto ke3 = [mpsGraph
          multiplicationWithPrimaryTensor:newCachedGraph->v3
                          secondaryTensor:newCachedGraph->v3 name:nil];
        auto ke = [mpsGraph additionWithPrimaryTensor:ke1 
                                      secondaryTensor:ke2 name:nil];
        ke = [mpsGraph additionWithPrimaryTensor:ke
                                 secondaryTensor:ke3 name:nil];
        ke = [mpsGraph multiplicationWithPrimaryTensor:ke
                                       secondaryTensor:half name:@"ke"];

        // pressure
        pres = [mpsGraph 
          divisionWithPrimaryTensor:newCachedGraph->etol
                    secondaryTensor:newCachedGraph->rho name:nil];
        pres = [mpsGraph 
          subtractionWithPrimaryTensor:pres
                       secondaryTensor:ke name:nil];
        pres = [mpsGraph
          divisionWithPrimaryTensor:pres
                    secondaryTensor:newCachedGraph->gm1 name:@"pres"];
      });

    // Output as placeholder
    Placeholder primPlaceholder = Placeholder(cachedGraph->prim, prim);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      cachedGraph->rho: getMPSGraphTensorData(nullptr, stream, cons.select(0, 0)),
      cachedGraph->m1: getMPSGraphTensorData(nullptr, stream, cons.select(0, 1)),
      cachedGraph->m2: getMPSGraphTensorData(nullptr, stream, cons.select(0, 2)),
      cachedGraph->m3: getMPSGraphTensorData(nullptr, stream, cons.select(0, 3)),
      cachedGraph->etol: getMPSGraphTensorData(nullptr, stream, cons.select(0, 4)),
      cachedGraph->gm1 : getMPSGraphTensorFromScalar(stream, gammad - 1),
    }

    // execute the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, primPlaceholder);
  }
}

} // namespace mps

// APIs exposed to at::native scope
TORCH_IMPL_FUNC(euler_cons2prim_out_mps)
(const Tensor & cons, double gammad, Tensor& prim) {
  TORCH_CHECK(cons.is_contiguous(), "cons must be contiguous");
  TORCH_CHECK(gammad > 0, "gamma must be positive");

  mps::cons2prim_mps(cons, prim, gammad);
}

} // namespace at::native
