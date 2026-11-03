#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The model docs generator's size columns and tools line, on fixed folder declarations (no catalog needed)."""
import importlib.util,pathlib,unittest
spec=importlib.util.spec_from_file_location('model_docs',pathlib.Path(__file__).with_name('generate-model-docs.py'));docs=importlib.util.module_from_spec(spec);spec.loader.exec_module(docs)
def weights(dtype,size):return {'dtype':dtype,'files':['model.safetensors'],'bytes':{'model.safetensors':size}}
class SizeTests(unittest.TestCase):
    def test_fp32_source_cast_to_fp16_names_the_download_and_the_smaller_copy(self):
        rife={'weights':weights('fp32',22_674_665),'derive':{'fp16':'cast:restoration-rife'}}
        self.assertEqual(docs.precision_sizes(rife,'fp16'),('22674665','~11337332'))
    def test_derived_tiers_do_not_repeat_the_16_bit_download(self):
        ace={'weights':weights('fp16',10_091_986_191),'derive':{'int8':'mlx-affine','int4':'mlx-affine'}}
        self.assertEqual(docs.precision_sizes(ace,'fp16'),('10091986191','10091986191'))
        self.assertEqual(docs.precision_sizes(ace,'int8'),('made from the 16-bit download','~5361367663'))
        self.assertEqual(docs.precision_sizes(ace,'int4'),('made from the 16-bit download','~2838371116'))
    def test_vendor_tier_and_declared_size_are_their_own(self):
        m={'weights':weights('bf16',100),'vendor':{'int4':weights('int4',40)},'format':[{'id':'bf16','size_bytes':90}]}
        self.assertEqual(docs.precision_sizes(m,'int4'),('40','40'))
        self.assertEqual(docs.precision_sizes(m,'bf16'),('100','90'))
    def test_unknown_sizes_stay_unknown(self):
        m={'weights':{'dtype':'fp16','files':['a','b'],'bytes':{'a':1}},'derive':{'int4':'mlx-affine'}}
        self.assertEqual(docs.precision_sizes(m,'fp16'),('unknown','unknown'))
        self.assertEqual(docs.precision_sizes(m,'int4'),('made from the 16-bit download','unknown'))
class ToolsTests(unittest.TestCase):
    def test_generic_window_is_named_plainly(self):
        self.assertEqual(docs.tools({}),'the generic window built from its operations')
        self.assertEqual(docs.tools({'tools':['operation']}),'the generic window built from its operations')
        self.assertEqual(docs.tools({'tools':['sound','operation']}),'sound, the generic window built from its operations')
class DefaultDownloadTests(unittest.TestCase):
    def pinned(self,dtype,**files):return {'dtype':dtype,'files':list(files),'bytes':files,'sha256':{f:f'{f}-hash' for f in files}}
    def test_a_vendor_tier_is_the_default_when_the_base_has_no_runner(self):
        fish={'weights':self.pinned('bf16',a=72),'vendor':{'int4':self.pinned('int4',b=27)},'runner':[{'name':'native'}],
              'format':[{'id':'bf16','runners':[]},{'id':'int4','runners':['native']}]}
        self.assertEqual(docs.default_precision(fish),'int4')
        self.assertEqual(docs.download_bytes(fish,'int4'),27)
    def test_assets_count_once_and_derived_tiers_download_the_base(self):
        m={'weights':self.pinned('fp32',a=10),'derive':{'fp16':'cast:x','int4':'mlx-affine'},'runner':[{'name':'native'}],
           'asset':[{'id':'enc','files':['e'],'bytes':{'e':5},'sha256':{'e':'e-hash'}},{'id':'dup','files':['a'],'bytes':{'a':10},'sha256':{'a':'a-hash'}}]}
        self.assertEqual(docs.default_precision(m),'fp16')
        self.assertEqual(docs.download_bytes(m,'int4'),15)
    def test_a_missing_pin_is_unknown_not_zero(self):
        m={'weights':{'dtype':'fp16','files':['a'],'bytes':{'a':1}}}
        self.assertIsNone(docs.download_bytes(m,'fp16'))
        self.assertEqual(docs.gigabytes(None),'unknown')
    def test_sizes_use_the_apps_decimal_units(self):
        self.assertEqual(docs.gigabytes(364_050_539_022),'364.1 GB')
        self.assertEqual(docs.gigabytes(440_000_000),'440 MB')
if __name__=='__main__':unittest.main()
