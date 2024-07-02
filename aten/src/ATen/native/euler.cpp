// Perform the conversion from conservative to primitive variables

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/cons2prim_native.h>
#endif

namespace at::native {
  TORCH_API at::Tensor & euler_cons2prim_cpu_out(
      const at::Tensor & cons, double gammad, at::Tensor & prim) 
  {
    TORCH_CHECK(cons.is_contiguous(), "cons must be contiguous");
    TORCH_CHECK(prim.is_contiguous(), "prim must be contiguous");

    auto cons_sizes = cons.sizes();

    TORCH_CHECK(cons_sizes[0] >= 5, "cons must have at least 5 fields");

    auto cons_data = cons.data_ptr<float>();
    auto prim_data = prim.data_ptr<float>();

    size_t ncells = 1;
    for (size_t i = 1; i < cons_sizes.size(); i++) {
      ncells *= cons_sizes[i];
    }

    auto cons_strides = cons.strides();
    int64_t nstride = cons_strides[0];

    for (size_t i = 0; i < ncells; i++) {
      float rho = cons_data[i * nstride + 0];
      float momx = cons_data[i * nstride + 1];
      float momy = cons_data[i * nstride + 2];
      float momz = cons_data[i * nstride + 3];
      float energy = cons_data[i * nstride + 4];

      prim_data[i * nstride + 0] = rho;
      prim_data[i * nstride + 1] = momx / rho;
      prim_data[i * nstride + 2] = momy / rho;
      prim_data[i * nstride + 3] = momz / rho;

      float ke = 0.5 * (momx * momx + momy * momy + momz * momz) / rho;
      prim_data[i * nstride + 4] = (energy - ke) / (gammad - 1.);
    }

    return prim;
  }

  TORCH_API at::Tensor euler_cons2prim_cpu(const at::Tensor & cons, double gammad)
  {
    TORCH_CHECK(cons.is_contiguous(), "cons must be contiguous");

    Tensor prim = at::empty(cons.sizes(), cons.options());
    euler_cons2prim_cpu_out(cons, gammad, prim);

    return prim;
  }

}
