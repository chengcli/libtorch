// Perform the conversion from conservative to primitive variables

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/cons2prim_native.h>
#endif

namespace at::native {
  TORCH_API at::Tensor & euler_cons2prim_cpu_out(
      const at::Tensor & cons, at::Tensor & prim) {
    TORCH_CHECK(cons.is_contiguous(), "cons must be contiguous");

    auto cons_sizes = cons.sizes();
    auto cons_strides = cons.strides();
    auto cons_data = cons.data_ptr<float>();

    auto prim_data = prim.data_ptr<float>();

    int64_t n_cells = cons_sizes[0];
    int64_t n_fields = cons_sizes[1];

    for (int64_t i = 0; i < n_cells; i++) {
      float rho = cons_data[i * cons_strides[0] + 0];
      float mom_x = cons_data[i * cons_strides[0] + 1];
      float mom_y = cons_data[i * cons_strides[0] + 2];
      float mom_z = cons_data[i * cons_strides[0] + 3];
      float energy = cons_data[i * cons_strides[0] + 4];

      prim_data[i * cons_strides[0] + 0] = rho;
      prim_data[i * cons_strides[0] + 1] = mom_x / rho;
      prim_data[i * cons_strides[0] + 2] = mom_y / rho;
      prim_data[i * cons_strides[0] + 3] = mom_z / rho;
      prim_data[i * cons_strides[0] + 4] = (energy - 0.5 * (mom_x * mom_x + mom_y * mom_y + mom_z * mom_z) / rho) / rho;
    }

    return prim;
  }

  TORCH_API at::Tensor euler_cons2prim_cpu(const at::Tensor & cons) {
    TORCH_CHECK(cons.is_contiguous(), "cons must be contiguous");

    auto prim = at::empty(cons.sizes(), cons.options());
    euler_cons2prim_cpu_out(cons, prim);

    return prim;
  }

}
