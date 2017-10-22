''' This script is used to auto-generate header. '''

import argparse
import CppHeaderParser
from os.path import join


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


def get_clean_type(name):
    clean = name.replace('const', '').replace('&', '').replace('boost::uint',
        'uint').strip()
    return None if clean == 'void' else clean


def get_overload_func(funcs):

    func_name = funcs[0]['name']

    body = 'PyObject *Uhd_{}(Uhd *self, PyObject *args) {{\n\n'.format(
           func_name)
    body += '    const Py_ssize_t nargs = PyTuple_Size(args);\n'

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

        body += '    ' + prefix + 'if (nargs >= {} && nargs <= {}'.format(
            len(required), len(required) + len(optional))

        conditions = []
        for index, param in enumerate(required):
            conditions.append('\n        && is<{}>(PyTuple_GetItem(args, {}))'
                .format(param['type'],index))
        for index, param in enumerate(optional):
            conditions.append('\n        && (nargs <= {} || is<{}>(PyTuple_Get'
                'Item(args, {})))'.format(index + len(required), param['type'],
                index + len(required)))

        body += ''.join(conditions) + ') {\n'
        body += '        return _{}_{}(self, args);\n'.format(func_name,
                func_index)
        prefix = '} else '
    body += '    }\n'
    body += '    return _{}_{}(self, args);\n}}\n'.format(func_name, 0)

    bodies = []
    for func_index, func in enumerate(funcs):
        func_body = get_func(func, '_{}_{}'.format(func_name, func_index),
                    'static ')
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

    body = '{}PyObject *{}(Uhd *self, PyObject *args) {{\n\n'.format(
           func_prefix, func_name)
    body += '    const Py_ssize_t nargs = PyTuple_Size(args);\n'
    body += '    if (nargs < {} || nargs > {})\n'.format(nrequired,
            nrequired + noptional)
    body += '        return PyErr_Format(PyExc_TypeError, "Invalid number of ' \
            'arguments: got %ld, expected {}.", nargs);\n\n'.format('[{}...{}]'
            .format(nrequired, nrequired + noptional) if noptional else
            str(nrequired) if nrequired else 'None')

    for index, param in enumerate(required):
        body += '    Expect<{}> {};\n'.format(param['type'], param['name'])
        body += '    if (!({} = to<{}>(PyTuple_GetItem(args, {}))))\n'.format(
                param['name'], param['type'], index)
        body += '        return PyErr_Format(PyExc_TypeError, "Invalid type' \
                ' for argument # {}: %s", {}.what());\n'.format(index + 1,
                param['name'])
    body += '\n' if nrequired else ''

    for param in optional:
        param_default = None
        body += '    Expect<{}> {}{};\n'.format(param['type'], param['name'],
                ' = {}'.format(param_default) if param_default else '')
    body += '\n' if noptional > 1 else ''

    for index, param in enumerate(optional):
        body += '    if (nargs > {} && !({} = to<{}>(PyTuple_GetItem(args, {}' \
                '))))\n'.format(index + nrequired, param['name'], param['type'],
                index + nrequired)
        body += '        return PyErr_Format(PyExc_TypeError, "Invalid type ' \
                'for argument # {}: %s", {}.what());\n'.format(
                index + nrequired + 1, param['name'])
    body += '\n' if noptional else ''

    ret_prefix = ''
    if ret_type:
        ret_default = None
        body += '    {} ret{};\n'.format(ret_type, ' = {}'.format(ret_default)
                if ret_default else '')
        ret_prefix = 'ret = '

    body += '    try {\n'
    body += '        std::lock_guard<std::mutex> lg(self->dev_lock);\n'

    prefix = ''
    nargs = [i + nrequired for i in range(noptional + 1)]
    if nargs and len(nargs) > 1:
        for narg in reversed(nargs):
            if narg == nargs[0]:
                body += '       {}\n'.format(prefix)
            else:
                body += '       {} if (nargs == {})\n'.format(prefix, narg)
            body += '            {}self->dev->{}({});\n'.format(ret_prefix,
                    func['name'], ', '.join([p['name'] + '.get()' for p in
                    (required + optional)[:narg]]))
            prefix = ' else'
    else:
        narg = nargs[0]
        body += '        {}self->dev->{}({});\n'.format(ret_prefix,
            func['name'], ', '.join([p['name'] + '.get()' for p in
            (required + optional)[:narg]]))

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


def generate(source, destination, suffix=''):

    # Get UHD_VERSION from <source>/version.hpp
    version_hpp = CppHeaderParser.CppHeader(join(source,
        'version.hpp'))
    version = [s for s in version_hpp.defines if
               'UHD_VERSION ' in s][0].split(' ')[-1]

    # Get API from <source>/multi_usrp.hpp
    multi_usrp_hpp = CppHeaderParser.CppHeader(join(source,
        'usrp', 'multi_usrp.hpp'))

    funcs = {}
    for func in multi_usrp_hpp.classes['multi_usrp']['methods']['public']:
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
            print('Skipped function {}: unknown param type: {}'.format(
                  func['name'], ', '.join([t for t in param_types if t not in
                  known_param_types])))
        elif ret_type not in known_ret_types and ret_type:
            print('Skipped function {}: unknown return type: {}'.format(
                  func['name'], ret_type))
        else:
            if func['name'] not in funcs:
                funcs[func['name']] = [func]
            else:
                funcs[func['name']].append(func)

    func_bodies = []

    # Iterate over all functions in alphabetical order: this will minimize
    # diffs between past/future output products
    for func_name in sorted(funcs.keys()):
        func = funcs[func_name]
        if len(func) == 1:
            func_body = get_func(func[0])
        else:
            func_body = get_overload_func(func)
        if func_body is None:
            raise ValueError('Function body was None.')
        func_bodies.append(func_body)

    body = '/** This file is automatically generated. ' \
         + 'Do not edit. **/\n\n' \
         + '#include <uhd/version.hpp>\n\n' \
         + '#if UHD_VERSION == {}\n\n'.format(version) \
         + '#ifndef __UHD_GEN_HPP__\n' \
         + '#define __UHD_GEN_HPP__\n\n' \
         + '#include <Python.h>\n\n' \
         + '#include <uhd/usrp/multi_usrp.hpp>\n' \
         + '#include <uhd/exception.hpp>\n' \
         + '#include <uhd/types/dict.hpp>\n\n' \
         + '#include "uhd.hpp"\n' \
         + '#include "uhd_types.hpp"\n\n' \
         + 'namespace uhd {\n\n'

    body += '\n'.join(func_bodies) + '\n' if len(func_bodies) else ''
    body += 'const std::vector<PyMethodDef> Uhd_gen_methods {\n'
    for func_name in sorted(funcs.keys()):
        body += '    {{\"{}\", (PyCFunction)Uhd_{}, METH_VARARGS, \"\"}},' \
                    '\n'.format(func_name, func_name)
    body += '};\n\n' \
              + '}\n\n' \
              + '#endif /** __UHD_GEN_HPP__ **/\n\n' \
              + '#endif /** UHD_VERSION **/\n'

    with open(join(destination, 'uhd_{}{}.hpp'.format(version, suffix)), 'w') as f:
        f.write(body)

    return version, body


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('source', type=str, help='Source header path.')
    parser.add_argument('destination', type=str,
        help='Destination header path.')
    args = parser.parse_args()
    generate(args.source, args.destination)


if __name__ == '__main__':
    main()
