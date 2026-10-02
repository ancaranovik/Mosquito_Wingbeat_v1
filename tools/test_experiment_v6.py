"""Exp010 masking and single-run artifact tests; optimization/inference are blocked."""
import ast
import contextlib
from copy import deepcopy
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import experiment_training as baseline
import experiment_training_v4 as v4
import experiment_training_v6 as v6
import test_experiment_v2 as fixture
from cache_consumer import (CLASSES, COUNTS, TRAIN_COUNTS, MANIFEST, SPLIT,
                            digest_json, read_json, sha256, write_json)
from experiment_runner import run, run_plan, training_module, validate_config
from experiment_review import verify_experiment, compare_validation_to_reference
from project_paths import REPO_ROOT, ProjectPaths

TEMPLATE = REPO_ROOT/'configs/experiments/time_mask_v6.json'


class V6Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v6_test_', dir=REPO_ROOT/'_local_only')
        self.storage = Path(self.temp.name)
        assert self.storage.resolve().is_relative_to((REPO_ROOT/'_local_only').resolve())
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {'EDGEAI_DATA_ROOT': str(self.storage)})
        env.start(); self.addCleanup(env.stop)
        for rel in [MANIFEST, SPLIT]:
            p=ProjectPaths.from_env().resolve(rel);p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REPO_ROOT/rel,p)

    def test_single_run_and_only_augmentation_differs(self):
        config=read_json(TEMPLATE)
        self.assertEqual(run_plan(config), [{'frontend':'03A','model':'ds_cnn','run_id':'03A_ds_cnn_seed42'}])
        self.assertIs(training_module(config),v6)
        before=v4.effective_config(read_json(REPO_ROOT/'configs/experiments/power075_v4.json')['training_overrides'])
        after=v6.effective_config(config['training_overrides'])
        self.assertEqual({k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)},
                         {'protocol_version','augmentation','augmentation_config'})
        self.assertEqual(after['weight_decay'],0)
        self.assertFalse(after['evaluate_test'])
        for changes in [{'frontends':['03A','03B']},{'models':['small_cnn']},{'evaluate_test':True},
                        {'seed':43},{'reference_experiment_id':'exp_008_power075_wd1e4'},
                        {'training_overrides':{'learning_rate':.0003}},
                        {'training_overrides':{'class_weight_exponent':.5}}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_config({**config,**changes})
        self.assertEqual(len(run_plan(read_json(REPO_ROOT/'configs/experiments/power075_v4.json'))),8)

    def test_mask_is_train_only_private_contiguous_and_reproducible(self):
        meta=pd.DataFrame({'split':['train','validation','test'],'label':[0,1,2]})
        features=np.full((3,96,64),6,dtype=np.float32);features.setflags(write=False)
        original=features.copy();norm={'kind':'train_global_zscore','mean':2.,'std':4.}
        a=v6.FeatureDataset(features,meta,'train',norm,augment=True)
        b=v6.FeatureDataset(features,meta,'train',norm,augment=True)
        rng=torch.get_rng_state().clone();masked=0
        for _ in range(100):
            x,_=a[0];y,_=b[0];self.assertTrue(torch.equal(x,y))
            zeros=x[0].eq(0).all(1).nonzero().flatten().tolist()
            if zeros:
                masked+=1;self.assertTrue(1<=len(zeros)<=8)
                self.assertEqual(zeros,list(range(zeros[0],zeros[0]+len(zeros))))
            self.assertTrue(bool(((x==0)|(x==1)).all()))
        self.assertTrue(10<masked<90)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for split in ['train','validation','test']:
            ds=v6.FeatureDataset(features,meta,split,norm)
            for _ in range(3):self.assertTrue(bool(ds[0][0].eq(1).all()))
        with self.assertRaises(RuntimeError):v6.FeatureDataset(features,meta,'validation',norm,augment=True)
        with self.assertRaises(RuntimeError):v6.FeatureDataset(features,meta,'test',norm,augment=True)
        np.testing.assert_array_equal(original,features)
        self.assertFalse(features.flags.writeable)

    def test_shuffle_unchanged_and_eval_loader_never_masks(self):
        meta=pd.DataFrame({'split':['train']*100+['validation'],'label':np.arange(101)})
        features=np.ones((101,96,64),dtype=np.float32)
        norm={'kind':'train_global_zscore','mean':0.,'std':1.}
        a=v6.make_loader(features,meta,'train',norm,shuffle=True)
        b=v4.make_loader(features,meta,'train',norm,shuffle=True)
        self.assertEqual([int(y) for _,ys in a for y in ys],[int(y) for _,ys in b for y in ys])
        for split in ['train','validation']:
            loader=v6.make_loader(features,meta,split,norm)
            self.assertFalse(loader.dataset.augment)
            self.assertTrue(all(bool(x.eq(1).all()) for x,_ in loader))

    def test_scientific_functions_unchanged(self):
        def nodes(file):
            return {n.name:ast.dump(n,include_attributes=False) for n in
                    ast.parse((REPO_ROOT/'tools'/file).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=nodes('experiment_training_v4.py'),nodes('experiment_training_v6.py')
        self.assertEqual(set(a),set(b))
        for name in set(a)-{'effective_config','FeatureDataset','make_loader','protocol_record','run_branch','validate_saved_run'}:
            self.assertEqual(a[name],b[name],name)
        self.assertEqual(a['fit_validation_checkpoint'],b['fit_validation_checkpoint'])

    def test_notebook_preview_and_train_command_one_run(self):
        n=read_json(REPO_ROOT/'notebooks/current/04D_experiments.ipynb')
        s=next(''.join(c['source']) for c in n['cells'] if 'EXPERIMENT_CONFIG =' in ''.join(c['source']))
        scope={'paths':ProjectPaths.from_env(),'repo':REPO_ROOT,'sys':__import__('sys'),'subprocess':__import__('subprocess')}
        with patch('subprocess.run') as command,contextlib.redirect_stdout(io.StringIO()):
            exec(s,scope);command.assert_not_called()
            self.assertFalse(scope['RUN_TRAINING']);self.assertIs(scope['engine'],v6)
            self.assertEqual(len(scope['plan']),1)
            exec(s.replace('RUN_TRAINING = False','RUN_TRAINING = True'),scope)
            self.assertEqual(command.call_args.args[0][5],'--config')
            self.assertIn('configs/experiments/time_mask_v6.json',command.call_args.args[0])

    def synthetic(self,engine,branch):
        with patch.object(fixture,'v2',engine):fixture.V2Tests.write_synthetic_run(self,branch)
        protocol=read_json(engine.OUTPUT/'protocol.json')
        accepted=deepcopy(protocol['class_weights']);protocol['accepted_class_weights']=accepted
        protocol['class_weights']=engine.effective_class_weights({'class_weights':accepted})
        for family in engine.SELECTED_FAMILIES:
            d=engine.RUNS/f'{branch}_{family}_seed42';r=read_json(d/'result.json')
            norm={'kind':'train_global_zscore','fit_split':'train','mean':0.,'std':1.,'train_examples':22613}
            write_json(d/'normalization.json',norm)
            write_json(d/'class_weights.json',engine.class_weight_record(engine.CONFIG,accepted))
            write_json(d/'run_spec.json',{'protocol':protocol})
            r.update(normalization=norm,class_weights=protocol['class_weights'],accepted_class_weights=accepted,
                     protocol_sha256=digest_json(protocol))
            files=set(r['artifacts_sha256'])|{'class_weights.json'}
            r['artifacts_sha256']={name:sha256(d/name) for name in files}
            write_json(d/'result.json',r)
        write_json(engine.OUTPUT/'protocol.json',protocol)

    def execute(self,engine,config,name):
        reference={'source_training_configuration':v4.effective_config(read_json(
            REPO_ROOT/'configs/experiments/power075_v4.json')['training_overrides'])}
        with patch('experiment_runner.smoke',return_value={'runtime':{'gpu':'MOCK'},'cache_sha256':{},'split_sha256':'MOCK'}), \
             patch('experiment_runner.subprocess.check_output',side_effect=['mock-commit','']), \
             patch('bounded_aggregation.authenticate_reference',return_value=(None,reference,{})), \
             patch.object(engine,'run_branch',side_effect=lambda b:self.synthetic(engine,b)), \
             patch.object(engine,'fit_validation_checkpoint',side_effect=AssertionError('Training forbidden')), \
             patch.object(engine,'save_evaluation',side_effect=AssertionError('Inference forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            return run(config,experiment_id=name)

    def test_single_run_roundtrip_reference_tamper_and_notebook(self):
        self.execute(v4,REPO_ROOT/'configs/experiments/power075_v4.json','exp_007_power075')
        previous=(v6.OUTPUT,v6.RUNS,v6.CONFIG,v6.SELECTED_FAMILIES,v6.EXPERIMENT_ID)
        target=self.execute(v6,TEMPLATE,'exp_v6_mock')
        self.assertEqual(previous,(v6.OUTPUT,v6.RUNS,v6.CONFIG,v6.SELECTED_FAMILIES,v6.EXPERIMENT_ID))
        meta=read_json(target/'config/metadata.json')
        self.assertEqual((meta['status'],meta['expected_runs'],meta['completed_runs']),('COMPLETE',1,1))
        self.assertEqual(len(verify_experiment('exp_v6_mock')),1)
        table=compare_validation_to_reference('exp_v6_mock','exp_007_power075')
        self.assertEqual(len(table),1)
        self.assertEqual(table.iloc[0]['Delta val F1 (percentage points)'],0)
        self.assertFalse(list(target.glob('results/runs/*/test*predictions.csv')))
        n=read_json(REPO_ROOT/'notebooks/current/04C_compare_models.ipynb')
        s=next(''.join(c['source']) for c in n['cells'] if 'all_complete = False' in ''.join(c['source']))
        s=s.replace('EXPERIMENT_NAME = "exp_010_time_mask"','EXPERIMENT_NAME = "exp_v6_mock"')
        scope={'PATHS':ProjectPaths.from_env(),'BRANCHES':v6.BRANCHES,'FAMILIES':v6.FAMILIES,
               'read_json':read_json,'pd':pd,'display':lambda _:None}
        with contextlib.redirect_stdout(io.StringIO()):exec(s,scope)
        self.assertTrue(scope['all_complete']);self.assertEqual(len(scope['comparison']),1)
        plot=next(''.join(c['source']) for c in n['cells'] if 'labels_short =' in ''.join(c['source']))
        import matplotlib
        matplotlib.use('Agg')
        with patch('matplotlib.pyplot.show') as show,contextlib.redirect_stdout(io.StringIO()):
            exec(plot,scope)
        show.assert_called_once()
        import matplotlib.pyplot as plt
        plt.close('all')
        artifact=target/'results/runs/03A_ds_cnn_seed42/validation_predictions.csv'
        with artifact.open('ab') as stream:stream.write(b'tampered')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):verify_experiment('exp_v6_mock')

    def test_actual_branch_wiring_without_training_or_inference(self):
        norm={'kind':'train_global_zscore','fit_split':'train','mean':0.,'std':1.,'train_examples':22613}
        with patch.object(fixture,'v2',v6),patch.object(v6,'fit_normalization',return_value=norm):
            fixture.V2Tests.test_real_branch_wiring_has_no_test_loader_and_authenticates_outputs(self)

    def test_bad_reference_blocks_before_gpu_or_output(self):
        before=v4.effective_config(read_json(REPO_ROOT/'configs/experiments/power075_v4.json')['training_overrides'])
        before['learning_rate']=.0003
        with patch('bounded_aggregation.authenticate_reference',return_value=(None,{'source_training_configuration':before},{})), \
             patch('experiment_runner.smoke') as smoke,self.assertRaisesRegex(RuntimeError,'beyond'):
            run(TEMPLATE)
        smoke.assert_not_called()
        self.assertFalse(ProjectPaths.from_env().experiment('exp_010_time_mask').exists())


if __name__=='__main__':unittest.main()
