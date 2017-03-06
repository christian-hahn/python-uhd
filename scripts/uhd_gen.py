#!/usr/bin/env python3

''' This script is used to automatically generate uhd_gen.cpp. '''

import CppHeaderParser

known_param_types = {'bool': {'default': 'false'},
                     'uint8_t': {'default': '0'},
                     'uint32_t': {'default': '0'},
                     'uint64_t': {'default': '0'},
                     'double': {'default': '0.0'},
                     'std::string': {'default': None},
                     'size_t': {'default': '0'},
                     'std::complex<double>': {'default': None},
                     'uhd::usrp::subdev_spec_t': {'default': None},
                     'tune_request_t': {'default': None}}

known_ret_types = {'void': {'default': None},
                   'bool': {'default': 'false'},
                   'uint32_t': {'default': '0'},
                   'uint64_t': {'default': '0'},
                   'double': {'default': '0.0'},
                   'size_t': {'default': '0'},
                   'std::string': {'default': None},
                   'tune_result_t': {'default': None},
                   'uhd::usrp::subdev_spec_t': {'default': None},
                   'std::vector<std::string>': {'default': None},
                   'dict<std::string, std::string>': {'default': None},
                   'meta_range_t': {'default': None},
                   'freq_range_t': {'default': None},
                   'gain_range_t': {'default': None}}

blacklist = ['make', 'get_device', 'get_rx_stream', 'get_tx_stream',
                 'issue_stream_cmd', 'get_rx_dboard_iface',
                 'get_tx_dboard_iface', 'set_clock_config']


class Text(object):
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'
    RESET   = '\033[39m'


def get_clean_type(name):
    clean = name.replace('const','').replace('&','').replace('boost::uint','uint').strip()
    return None if clean == 'void' else clean

def get_overload_func(funcs):

    func_name = funcs[0]['name']

    body = 'PyObject *Uhd_%s(Uhd *self, PyObject *args) {\n\n' % (func_name) \
         + '    const Py_ssize_t nargs = PyTuple_Size(args);\n'

    prefix = ''
    for func_index, func in enumerate(funcs):
        required = []
        optional = []
        for index, param in enumerate(func['parameters']):
            clean_type = get_clean_type(param['type'])
            if not clean_type:
                continue
            if 'defaultValue' in param:
                optional.append({'name':param['name'], 'type':clean_type})
            else:
                required.append({'name':param['name'], 'type':clean_type})

        body += '    ' + prefix + 'if (nargs >= %d && nargs <= %d' % (len(required),len(required)+len(optional))

        conditions = []
        for index, param in enumerate(required):
            conditions.append('\n        && is<%s>(PyTuple_GetItem(args, %d))' % (param['type'],index))
        for index, param in enumerate(optional):
            conditions.append('\n        && (nargs <= %d || is<%s>(PyTuple_GetItem(args, %d)))' % (index+len(required), param['type'],index+len(required)))

        body += ''.join(conditions) + ') {\n' \
             + '        return _%s_%d(self, args);\n' % (func_name,func_index)
        prefix = '} else '
    body += '    }\n' \
         + '    return _%s_%d(self, args);\n}\n' % (func_name,0)

    bodies = []
    for func_index, func in enumerate(funcs):
        func_body = get_func(func, '_%s_%s' % (func_name,func_index), 'static ')
        if func_body is None:
            raise ValueError('Function body was None.')
        bodies.append(func_body)

    body = '\n'.join(bodies) + '\n' + body

    return body


def get_func(func, func_name=None, func_prefix=''):
    if not func_name:
        func_name = 'Uhd_' + func['name']
    required = []
    optional = []
    for index, param in enumerate(func['parameters']):
        clean_type = get_clean_type(param['type'])
        if not clean_type:
            continue
        if 'defaultValue' in param:
            optional.append({'name':param['name'], 'type':clean_type})
        else:
            required.append({'name':param['name'], 'type':clean_type})
    nrequired = len(required)
    noptional = len(optional)

    ret_type = get_clean_type(func['rtnType'])

    body = '%sPyObject *%s(Uhd *self, PyObject *args) {\n\n' % (func_prefix, func_name) \
         + '    const Py_ssize_t nargs = PyTuple_Size(args);\n' \
         + '    if (nargs < %d || nargs > %d)\n' % (nrequired,nrequired+noptional) \
         + '        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments:' \
         + ' got %%ld, expected %s.", nargs);\n\n' % ('[%d...%d]' % (nrequired,nrequired+noptional) if \
                                                     noptional else str(nrequired) if nrequired else 'None')

    for index, param in enumerate(required):
        body += '    Expect<%s> %s;\n' % (param['type'],param['name']) \
              + '    if (!(%s = to<%s>(PyTuple_GetItem(args, %d))))\n' % (param['name'],param['type'],index) \
              + '        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # %d: %%s", %s.what());\n' % (index+1,param['name'])
    body += '\n' if nrequired else ''

    for param in optional:
        param_default = None#known_param_types[param['type']]['default']
        body += '    Expect<%s> %s%s;\n' % (param['type'],param['name'],' = %s' % param_default if param_default else '')
    body += '\n' if noptional > 1 else ''

    for index, param in enumerate(optional):
        body += '    if (nargs > %d && !(%s = to<%s>(PyTuple_GetItem(args, %d))))\n' % (index+nrequired,param['name'],param['type'],index+nrequired) \
              + '        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # %d: %%s", %s.what());\n' % (index+nrequired+1,param['name'])
    body += '\n' if noptional else ''

    ret_prefix = ''
    if ret_type:
        ret_default = None#known_ret_types[ret_type]
        body += '    %s ret%s;\n' % (ret_type, ' = %s' % ret_default if ret_default else '')
        ret_prefix = 'ret = '

    body += '    try {\n' \
          + '        std::lock_guard<std::mutex> lg(self->dev_lock);\n'

    prefix = ''
    nargs = [i+nrequired for i in range(noptional+1)]
    if nargs and len(nargs) > 1:
        for narg in reversed(nargs):
            if narg == nargs[0]:
                body += '       %s\n' % (prefix)
            else:
                body += '       %s if (nargs == %d)\n' % (prefix,narg)
            body += '            %sself->dev->%s(%s);\n' % (ret_prefix,func['name'],', '.join([p['name'] + '.get()' for p in (required+optional)[:narg]]))
            prefix = ' else'
    else:
        narg = nargs[0]
        body += '        %sself->dev->%s(%s);\n' % (ret_prefix,func['name'],', '.join([p['name'] + '.get()' for p in (required+optional)[:narg]]))

    body += '    } catch(const uhd::exception &e) {\n' \
         + '        return PyErr_Format(UhdError, "%s", e.what());\n' \
         + '    }\n\n'

    if ret_type:
        body += '    return from(ret);\n'
    else:
        body += '    Py_INCREF(Py_None);\n' \
             + '    return Py_None;\n'
    body += '}\n'

    return body


def main():

    header_filename = '/usr/include/uhd/usrp/multi_usrp.hpp'
    cpp = CppHeaderParser.CppHeader(header_filename)

    funcs = {}
    for func in cpp.classes['multi_usrp']['methods']['public']:
        param_types = []
        for index, param in enumerate(func['parameters']):
            clean_type = get_clean_type(param['type'])
            if clean_type:
                param_types.append(clean_type)
        ret_type = get_clean_type(func['rtnType'])

        if func['name'] in blacklist:
            print('Skipped function {}: blacklisted'.format(func['name']))
        elif func['constructor']:
            print('Skipped function {}: constructor'.format(func['name']))
        elif func['destructor']:
            print('Skipped function {}: destructor'.format(func['name']))
        elif any([t not in known_param_types for t in param_types]):
            print(Text.RED + 'Skipped function {}: unknown param type: {}'.format(func['name'], \
                  ', '.join([t for t in param_types if t not in known_param_types])) + Text.RESET)
        elif ret_type not in known_ret_types and ret_type:
            print(Text.RED + 'Skipped function {}: unknown return type: {}'.format(func['name'], ret_type) + Text.RESET)
        else:
            if func['name'] not in funcs:
                funcs[func['name']] = [func]
            else:
                funcs[func['name']].append(func)

    func_bodies = []

    for func_name, func in funcs.items():
        if len(func) == 1:
            func_body = get_func(func[0])
        else:
            func_body = get_overload_func(func)
        if func_body is None:
            raise ValueError('Function body was None.')
        func_bodies.append(func_body)

    cpp_body = '#include <Python.h>\n\n' \
             + '#include <uhd/usrp/multi_usrp.hpp>\n' \
             + '#include <uhd/exception.hpp>\n' \
             + '#include <uhd/types/dict.hpp>\n\n' \
             + '#include "uhd.hpp"\n' \
             + '#include "uhd_types.hpp"\n' \
             + '#include "uhd_gen.hpp"\n\n' \
             + 'namespace uhd {\n\n'

    cpp_body += '\n'.join(func_bodies) + '\n' if len(func_bodies) else ''
    cpp_body += 'const std::vector<PyMethodDef> Uhd_gen_methods = {{\n'
    for func_name, func in funcs.items():
        cpp_body += '    {\"%s\", (PyCFunction)Uhd_%s, METH_VARARGS, \"\"},\n' % (func_name,func_name)
    cpp_body += '    }};\n' \
              + '};\n'

    with open('uhd_gen.cpp', 'w') as f:
        f.write(cpp_body)

    hpp_body = '#ifndef __UHD_GEN_HPP__\n' \
             + '#define __UHD_GEN_HPP__\n\n' \
             + '#include <Python.h>\n\n' \
             + 'namespace uhd {\n\n' \
             + 'extern const std::vector<PyMethodDef> Uhd_gen_methods;\n\n' \
             + '};\n\n' \
             + '#endif /** __UHD_GEN_HPP__ **/\n'

    with open('uhd_gen.hpp', 'w') as f:
        f.write(hpp_body)

if __name__ == '__main__':
    main()
