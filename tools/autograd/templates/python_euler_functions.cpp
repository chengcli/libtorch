#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_euler_functions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/structseq.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::TensorList;

using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef euler_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPEulerVariableFunctionsModule = NULL;

void initEulerFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._euler",
     NULL,
     -1,
     euler_functions
  };
  PyObject* euler = PyModule_Create(&def);
  THPEulerVariableFunctionsModule = euler;
  if (!euler) {
    throw python_error();
  }
  // steals a reference to euler
  if (PyModule_AddObject(module, "_euler", euler) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace torch::autograd
