/* Copyright SunnyCase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "env_vars.h"
#include "ffi_modules.h"
#include <algorithm>
#include <map>
#include <pybind11/stl.h>

using namespace nncase;

namespace {
bool str_eq_ignore_case(const char *s1, const char *s2, int n) {
    for (int i = 0; i < n; ++i) {
        if (tolower(s1[i]) != s2[i])
            return false;
    }
    return true;
}

int strlen_max(const char *str, int max) {
    for (int i = 0; i <= max; ++i) {
        if (str[i] == '\0') {
            return i;
        }
    }
    return 0;
}

bool is_truthy(char *str) {
    int len = strlen_max(str, 4);
    switch (len) {
    case 1:
        return str[0] == '1' || tolower(str[0]) == 'y';
    case 2:
        return str_eq_ignore_case(str, "on", len);
    case 3:
        return str_eq_ignore_case(str, "yes", len);
    case 4:
        return str_eq_ignore_case(str, "true", len);
    default:
        return false;
    }
}

PyObject *py_getenv([[maybe_unused]] PyObject *self, PyObject *const *args,
                    Py_ssize_t nargs) {
    if (!(nargs == 1 || nargs == 2)) {
        PyErr_SetString(PyExc_TypeError, "getenv expected 1 or 2 arguments");
        return NULL;
    }
    PyObject *name = args[0];
    PyObject *default_val = nargs == 2 ? args[1] : Py_None;
    if (!PyUnicode_CheckExact(name)) {
        PyErr_SetString(PyExc_TypeError, "name must be a string");
        return NULL;
    }
    char *env_val = getenv(PyUnicode_AsUTF8(name));
    if (!env_val) {
        Py_INCREF(default_val);
        return default_val;
    }
    return PyUnicode_FromString(env_val);
}

PyObject *py_getenv_bool([[maybe_unused]] PyObject *self, PyObject *const *args,
                         Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "getenv_bool expected 2 arguments");
        return NULL;
    }
    PyObject *name = args[0];
    PyObject *default_val = args[1];
    if (!PyUnicode_CheckExact(name)) {
        PyErr_SetString(PyExc_TypeError, "name must be a string");
        return NULL;
    }
    char *env_val = getenv(PyUnicode_AsUTF8(name));
    PyObject *res = default_val;
    if (env_val) {
        res = is_truthy(env_val) ? Py_True : Py_False;
    }
    Py_INCREF(res);
    return res;
}

static PyMethodDef ModuleMethods[] = {
    {"getenv", reinterpret_cast<PyCFunction>((void *)py_getenv), METH_FASTCALL,
     NULL},
    {"getenv_bool", reinterpret_cast<PyCFunction>((void *)py_getenv_bool),
     METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL} // sentinel
};

void assertIsRecognized(const std::string &env) {
    bool is_invalidating = CACHE_INVALIDATING_ENV_VARS.find(env.c_str()) !=
                           CACHE_INVALIDATING_ENV_VARS.end();
    bool is_neutral = CACHE_NEUTRAL_ENV_VARS.find(env.c_str()) !=
                      CACHE_NEUTRAL_ENV_VARS.end();
    std::string errmsg = env + "is not recognized. "
                               "Please add it to triton/tools/sys/getenv.hpp";
    assert((is_invalidating || is_neutral) && errmsg.c_str());
}

std::mutex getenv_mutex;

std::string getStrEnv(const std::string &env) {
    std::lock_guard<std::mutex> lock(getenv_mutex);
    assertIsRecognized(env);
    const char *cstr = std::getenv(env.c_str());
    if (!cstr)
        return "";
    std::string result(cstr);
    return result;
}

// return value of a cache-invalidating boolean environment variable
bool getBoolEnv(const std::string &env) {
    std::lock_guard<std::mutex> lock(getenv_mutex);
    assertIsRecognized(env);
    const char *s = std::getenv(env.c_str());
    std::string str(s ? s : "");
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return str == "on" || str == "true" || str == "1";
}

inline std::optional<bool> isEnvValueBool(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (str == "on" || str == "true" || str == "1")
        return true;
    if (str == "off" || str == "false" || str == "0")
        return false;
    return std::nullopt;
}
} // namespace

void nncase::init_triton_env_vars(py::module &m) {
    m.def("get_cache_invalidating_env_vars",
          []() -> std::map<std::string, std::string> {
              std::map<std::string, std::string> ret;
              for (const auto &envVar : CACHE_INVALIDATING_ENV_VARS) {
                  auto strVal = getStrEnv(envVar);
                  if (strVal.empty())
                      continue;
                  auto boolV = isEnvValueBool(strVal);
                  if (boolV.has_value())
                      ret[envVar] = boolV.value() ? "true" : "false";
                  else
                      ret[envVar] = strVal;
              }
              return ret;
          });
    PyModule_AddFunctions(m.ptr(), ModuleMethods);
}
