#!/usr/bin/env python3
"""Standalone bounded Vulkan AGX delivery baseline; no shared runner imports."""
import argparse
import ctypes as C
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ICD = Path('/usr/share/vulkan/icd.d/asahi_icd.aarch64.json')
DRIVER = Path('/usr/lib64/libvulkan_asahi.so')
DRIVER_HASH = '26abaa59964177c0acf1860ba19cfd4b97c25435a8c4d81f9052973bc6bf1c65'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def spirv(kind, grammar, directory):
    """Encode SPIR-V 1.3 by instruction schema, without compiler/assembler.

    IDs: void1,uint2,bool3,v3uint4,fn5,input-ptr6,gid7,array8,
    output-struct9,output-ptr10,output11,uint-output-ptr12,push-struct13,
    push-ptr14,push15,uint-push-ptr16,zero17,eight18,constant19,mult20,
    main21,label22,gid-value23,index24,count-ptr25,count26,condition27,
    body28,merge29,offset30,store-ptr31,product32,value33.
    """
    schema = json.loads(grammar.read_text())
    ops = {i['opname']: i for i in schema['instructions']}
    enums = {k['kind']: {e['enumerant']: int(e['value'], 0) if isinstance(e['value'], str) else e['value']
                         for e in k.get('enumerants', [])} for k in schema['operand_kinds']}
    def e(kind, name):
        return enums[kind][name]
    instructions = []
    words = [0x07230203, 0x00010300, 0, 34, 0]  # unregistered generator=0; bound=34
    def emit(name, *args):
        instructions.append({'opname': name, 'opcode': ops[name]['opcode'], 'operands': list(args)})
        words.extend([((len(args)+1)<<16) | ops[name]['opcode'], *args])
    emit('OpCapability', e('Capability', 'Shader'))
    emit('OpMemoryModel', e('AddressingModel', 'Logical'), e('MemoryModel', 'GLSL450'))
    emit('OpEntryPoint', e('ExecutionModel', 'GLCompute'), 21, 0x6e69616d, 0, 7)
    emit('OpExecutionMode', 21, e('ExecutionMode', 'LocalSize'), 32, 1, 1)
    emit('OpDecorate', 7, e('Decoration', 'BuiltIn'), e('BuiltIn', 'GlobalInvocationId'))
    emit('OpDecorate', 8, e('Decoration', 'ArrayStride'), 4)
    emit('OpDecorate', 9, e('Decoration', 'Block'))
    emit('OpMemberDecorate', 9, 0, e('Decoration', 'Offset'), 0)
    emit('OpDecorate', 11, e('Decoration', 'DescriptorSet'), 0)
    emit('OpDecorate', 11, e('Decoration', 'Binding'), 0)
    emit('OpDecorate', 13, e('Decoration', 'Block'))
    emit('OpMemberDecorate', 13, 0, e('Decoration', 'Offset'), 0)
    emit('OpTypeVoid', 1); emit('OpTypeInt', 2, 32, 0); emit('OpTypeBool', 3)
    emit('OpTypeVector', 4, 2, 3); emit('OpTypeFunction', 5, 1)
    emit('OpTypePointer', 6, e('StorageClass', 'Input'), 4)
    emit('OpVariable', 6, 7, e('StorageClass', 'Input'))
    emit('OpTypeRuntimeArray', 8, 2); emit('OpTypeStruct', 9, 8)
    emit('OpTypePointer', 10, e('StorageClass', 'StorageBuffer'), 9)
    emit('OpVariable', 10, 11, e('StorageClass', 'StorageBuffer'))
    emit('OpTypePointer', 12, e('StorageClass', 'StorageBuffer'), 2)
    emit('OpTypeStruct', 13, 2); emit('OpTypePointer', 14, e('StorageClass', 'PushConstant'), 13)
    emit('OpVariable', 14, 15, e('StorageClass', 'PushConstant'))
    emit('OpTypePointer', 16, e('StorageClass', 'PushConstant'), 2)
    for ident, value in [(17, 0), (18, 8), (19, 0x12345678), (20, 0x9e3779b9)]:
        emit('OpConstant', 2, ident, value)
    emit('OpFunction', 1, 21, 0, 5); emit('OpLabel', 22)
    emit('OpLoad', 4, 23, 7); emit('OpCompositeExtract', 2, 24, 23, 0)
    emit('OpAccessChain', 16, 25, 15, 17); emit('OpLoad', 2, 26, 25)
    emit('OpULessThan', 3, 27, 24, 26); emit('OpSelectionMerge', 29, 0)
    emit('OpBranchConditional', 27, 28, 29); emit('OpLabel', 28)
    emit('OpIAdd', 2, 30, 24, 18); emit('OpAccessChain', 12, 31, 11, 17, 30)
    if kind == 'index':
        emit('OpIMul', 2, 32, 24, 20); emit('OpBitwiseXor', 2, 33, 32, 19)
    emit('OpStore', 31, 33 if kind == 'index' else 19)
    emit('OpBranch', 29); emit('OpLabel', 29); emit('OpReturn'); emit('OpFunctionEnd')
    data = struct.pack('<'+'I'*len(words), *words)
    path = directory / (kind + '.spv'); path.write_bytes(data)
    dump(directory / (kind + '-encoding.json'), {'header': words[:5], 'instructions': instructions,
         'used_instruction_schema': [ops[n] for n in sorted({i['opname'] for i in instructions})],
         'operand_enums': enums, 'grammar_sha256': sha(grammar)})
    expression = '0x12345678u' if kind == 'constant' else '(i * 0x9e3779b9u) ^ 0x12345678u'
    (directory / (kind + '.comp')).write_text('#version 450\nlayout(local_size_x=32) in;\n'
        'layout(set=0,binding=0,std430) buffer Output { uint data[]; };\n'
        'layout(push_constant) uniform Params { uint count; };\n'
        f'void main() {{ uint i=gl_GlobalInvocationID.x; if(i<count) data[8u+i]={expression}; }}\n')
    return path


def validate(path):
    # Public SPIRV-Tools C ABI; parse the target name to avoid enum-number assumptions.
    lib = C.CDLL('/usr/lib64/libSPIRV-Tools-shared.so')
    lib.spvParseTargetEnv.argtypes = [C.c_char_p, C.POINTER(C.c_int)]; lib.spvParseTargetEnv.restype = C.c_bool
    lib.spvContextCreate.argtypes = [C.c_int]; lib.spvContextCreate.restype = C.c_void_p
    lib.spvContextDestroy.argtypes = [C.c_void_p]
    lib.spvValidateBinary.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.c_size_t, C.POINTER(C.c_void_p)]
    lib.spvValidateBinary.restype = C.c_int
    lib.spvDiagnosticPrint.argtypes = [C.c_void_p]; lib.spvDiagnosticDestroy.argtypes = [C.c_void_p]
    lib.spvSoftwareVersionDetailsString.restype = C.c_char_p
    target = C.c_int(); assert lib.spvParseTargetEnv(b'vulkan1.3', C.byref(target))
    context = lib.spvContextCreate(target); assert context
    data = path.read_bytes(); array = (C.c_uint32 * (len(data)//4)).from_buffer_copy(data); diag = C.c_void_p()
    result = lib.spvValidateBinary(context, array, len(array), C.byref(diag))
    if diag:
        lib.spvDiagnosticPrint(diag); lib.spvDiagnosticDestroy(diag)
    lib.spvContextDestroy(context)
    return {'result': result, 'target': 'vulkan1.3', 'version': lib.spvSoftwareVersionDetailsString().decode(), 'sha256': sha(path)}


def oracle(data, kind, count, control=False, constant=0x12345678):
    rounded = (count+31)//32*32
    expected = [0xcdcdcdcd]*(8+rounded+8) + [0xa5a50000+j for j in range(16)]
    if not control:
        for i in range(count):
            expected[8+i] = constant if kind == 'constant' else ((i*0x9e3779b9)&0xffffffff) ^ constant
    actual = list(struct.unpack('<'+'I'*(len(data)//4), data)) if len(data)%4 == 0 else []
    return {'passed': actual == expected, 'expected': expected, 'actual': actual}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    p.add_argument('--include', type=Path, default=Path('/save/workspace/AGX_Probe/mesa/include'))
    p.add_argument('--grammar', type=Path, default=Path('/save/workspace/AGX_Probe/mesa/src/compiler/spirv/spirv.core.grammar.json'))
    p.add_argument('--runtime', type=Path, default=Path('/scratch/apple-m2-playground/vulkan-runtime/usr/lib64'))
    p.add_argument('--validate', type=Path, help='internal bounded validation child')
    p.add_argument('--probe-mov', action='store_true', help='after baseline, patch one pinned AGX mov immediate and run controls')
    args = p.parse_args()
    if args.validate:
        result = validate(args.validate); print(json.dumps(result)); return int(result['result'] != 0)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output or Path('/scratch/apple-m2-playground/vulkan-compute-runs') / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {'passed': False, 'utc': stamp, 'commands': [], 'cases': []}
    prefixes = ('VK_', 'MESA_', 'ASAHI_', 'AGX_', 'HK_', 'LIBGL_', 'GALLIUM_', 'DRI_', 'LD_')
    env = {k:v for k,v in os.environ.items() if not k.startswith(prefixes)}
    overrides = {'VK_DRIVER_FILES': str(ICD), 'VK_ICD_FILENAMES': str(ICD),
                 'VK_LOADER_LAYERS_DISABLE': '~implicit~', 'LD_LIBRARY_PATH': str(args.runtime.resolve()),
                 'MESA_SHADER_CACHE_DISABLE': 'true'}
    child = env | overrides
    summary['removed_environment_keys'] = sorted(set(os.environ)-set(env)); summary['child_overrides'] = overrides
    def run(argv, name, timeout=15, child_env=env, success=True):
        directory = out/name; directory.mkdir()
        record = {'argv': [str(x) for x in argv], 'timeout_seconds': timeout}
        summary['commands'].append(record)
        try:
            result = subprocess.run(record['argv'], env=child_env, capture_output=True, timeout=timeout)
            stdout, stderr = result.stdout, result.stderr; record['returncode'] = result.returncode
        except subprocess.TimeoutExpired as e:
            stdout, stderr = e.stdout or b'', e.stderr or b''; record['timed_out'] = True; record['returncode'] = None
        (directory/'stdout.txt').write_bytes(stdout); (directory/'stderr.txt').write_bytes(stderr)
        dump(directory/'command.json', record)
        if success and record['returncode'] != 0:
            raise RuntimeError('command failed: '+name)
        return stdout, record
    try:
        sources = out/'sources'; sources.mkdir()
        for f in [REPO/'tests/vulkan_compute.c', Path(__file__), REPO/'scripts/patch_agx_shader.py']: shutil.copy2(f, sources/f.name)
        files = [DRIVER, ICD, Path('/usr/lib64/libvulkan.so.1'), args.runtime/'libdisplay-info.so.2', args.grammar,
                 REPO/'tests/vulkan_compute.c', Path(__file__), Path('/usr/lib64/libSPIRV-Tools-shared.so')]
        files += sorted(args.include.glob('vulkan/*.h')) + sorted(args.include.glob('vk_video/*.h'))
        summary['sha256'] = {str(f.resolve()): sha(f) for f in files}
        if sha(DRIVER) != DRIVER_HASH: raise RuntimeError('actual driver hash changed')
        run(shlex.split(args.cc)+['--version'], 'compiler')
        binary = out/'vulkan_compute'
        run(shlex.split(args.cc)+['-std=c11','-Wall','-Wextra','-Werror','-O2','-MD','-MF',out/'dependencies.d',
            '-I'+str(args.include.resolve()), REPO/'tests/vulkan_compute.c','-ldl','-o',binary], 'build',30)
        summary['executable_sha256'] = sha(binary)
        deptext = (out/'dependencies.d').read_text().replace('\\\n',' ')
        summary['build_dependency_hashes'] = {str(Path(f).resolve()):sha(f) for f in shlex.split(deptext.split(':',1)[1])}
        depout, _ = run(['ldd', DRIVER], 'dependencies', child_env=child)
        if b'not found' in depout: raise RuntimeError('unresolved runtime dependency')
        # Hash the resolved ICD dependency closure without extending capability inventory.
        deps = []
        for line in depout.decode().splitlines():
            fields=line.split()
            candidate=fields[2] if '=>' in fields and len(fields)>2 else fields[0] if fields else ''
            if candidate.startswith('/') and Path(candidate).is_file(): deps.append(Path(candidate))
        summary['runtime_dependency_hashes'] = {str(f.resolve()):sha(f) for f in deps}
        runtime_files = [DRIVER, Path('/usr/lib64/libvulkan.so.1'), args.runtime/'libdisplay-info.so.2'] + deps
        runtime_expected = {str(f): sha(f) for f in runtime_files}
        summary['runtime_expected'] = runtime_expected
        def verify_runtime():
            if {str(f):sha(f) for f in runtime_files} != runtime_expected:
                raise RuntimeError('runtime changed during suite')
        modules = {}
        for kind in ('constant','index'):
            modules[kind] = spirv(kind,args.grammar,out)
            stdout,_=run([sys.executable,Path(__file__),'--validate',modules[kind]],'validate-'+kind)
            summary[kind+'_validation']=json.loads(stdout)
        def case(name, mode, code, kind, n, constant=0x12345678):
            verify_runtime()
            input_hash=sha(code)
            readback=out/(name+'.readback'); exported=out/(kind+'.bin')
            stdout,record=run([binary,mode,code,n,'/dev/dri/renderD128',readback,exported],name,child_env=child)
            verify_runtime()
            events=[json.loads(line) for line in stdout.decode().splitlines()]
            if not events or events[-1].get('completed') is not True: raise RuntimeError('missing completion')
            check=oracle(readback.read_bytes(),kind,n,mode=='control',constant)
            check.update(name=name,mode=mode,count=n,expected_constant=constant,input_sha256=input_hash,readback_sha256=sha(readback),api_events=events)
            dump(out/(name+'-oracle.json'),check); summary['cases'].append(check)
            if not check['passed'] or sha(code)!=input_hash: raise RuntimeError('oracle/input changed: '+name)
            if mode=='export': summary[kind+'_binary']={'bytes':exported.stat().st_size,'sha256':sha(exported)}
        case('no-dispatch','control',modules['constant'],'constant',33)
        for kind in ('constant','index'):
            case(kind+'-source-33','export',modules[kind],kind,33)
            for n in (1,32,33,257): case(kind+'-import-'+str(n),'import',out/(kind+'.bin'),kind,n)
        # Fail closed on an existing wrong character device, before logical-device creation.
        stdout,record=run([binary,'import',out/'constant.bin',33,'/dev/null',out/'wrong.readback',out/'unused.bin'],
                          'wrong-node',child_env=child,success=False)
        if record['returncode'] != 1 or b'vkCreateDevice' in stdout or (out/'wrong.readback').exists():
            raise RuntimeError('negative identity control did not fail before device creation')
        summary['wrong_node_control_passed']=True
        # This software control checks that a no-dispatch sentinel cannot satisfy the payload oracle.
        summary['no_dispatch_rejected_as_dispatch']=not oracle((out/'no-dispatch.readback').read_bytes(),'constant',33)['passed']
        if not summary['no_dispatch_rejected_as_dispatch']: raise RuntimeError('oracle control failed')
        if args.probe_mov:
            patch_dir=out/'mov-patch'
            run([sys.executable,REPO/'scripts/patch_agx_shader.py',out/'constant.bin','--output',patch_dir], 'patcher')
            case('mov-before','import',out/'constant.bin','constant',33)
            case('mov-patched-33','import',patch_dir/'patched.bin','constant',33,0x12345679)
            case('mov-patched-257','import',patch_dir/'patched.bin','constant',257,0x12345679)
            case('mov-after','import',out/'constant.bin','constant',33)
            stdout,record=run([binary,'import',patch_dir/'stale-integrity.bin',33,'/dev/dri/renderD128',
                              out/'stale.readback',out/'unused.bin'], 'stale-integrity',child_env=child,success=False)
            events=[json.loads(line) for line in stdout.decode().splitlines()]
            create=[e for e in events if e.get('api')=='vkCreateShadersEXT']
            if (record['returncode']!=1 or len(create)!=1 or create[0]['result']!=1000482000 or
                any(e.get('api')=='vkQueueSubmit' for e in events) or (out/'stale.readback').exists()):
                raise RuntimeError('stale integrity not rejected before submission')
            summary['stale_integrity_control']={'passed':True,'api_events':events,'sha256':sha(patch_dir/'stale-integrity.bin')}
            summary['instruction_patch']=json.loads((patch_dir/'patch.json').read_text())
        verify_runtime()
        summary['runtime_unchanged_before_after_each_case']=True
        summary['passed']=True
    except Exception as e:
        summary['error']=f'{type(e).__name__}: {e}'
    finally:
        dump(out/'record.json',summary)
        print(json.dumps({'passed':summary['passed'],'output':str(out),'error':summary.get('error')}))
    return 0 if summary['passed'] else 1

if __name__ == '__main__': sys.exit(main())
