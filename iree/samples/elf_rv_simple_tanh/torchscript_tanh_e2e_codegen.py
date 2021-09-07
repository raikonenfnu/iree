# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

import npcomp
from npcomp.passmanager import *
from npcomp.compiler.pytorch.backend import iree, frontend_lowering
from npcomp.compiler.utils import logging
import iree.compiler as ireec

import os

logging.enable()

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

def compile_to_mlir(imported_module):
    with imported_module.context as context:
        pipeline_str = "npcomp-backend-to-iree-frontend-pipeline"
        pm = PassManager.parse(pipeline_str)
        pm.run(imported_module)
    return str(imported_module)

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tanh(x)

test_module = TestModule()
class_annotator = torch_mlir.ClassAnnotator()
recursivescriptmodule = torch.jit.script(test_module)
torch.jit.save(recursivescriptmodule, '/tmp/foo.pt')

class_annotator.exportNone(recursivescriptmodule._c._type())
class_annotator.exportPath(recursivescriptmodule._c._type(), ['forward'])
class_annotator.annotateArgs(recursivescriptmodule._c._type(), ['forward'], [
    None,
    ([4], torch.float32, True)
])
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator)
#mb.module.operation.print()

compiled = compile_to_mlir(frontend_lowering.lower_object_graph(mb.module))
ARITFACTS_DIR = "/tmp"
mlir_path = os.path.join(ARITFACTS_DIR, "tanh.mlir")
with open(mlir_path, "wt") as output_file:
    output_file.write(compiled)