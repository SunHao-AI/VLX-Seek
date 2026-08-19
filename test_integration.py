#!/usr/bin/env python3
"""Integration test for multi-GPU pseudo-label generation with queue mode."""

import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import multiprocessing as mp

# Add current directory to path
import sys
sys.path.insert(0, '.')

from distill.generate_pseudo_labels import run_multigpu, _worker_shard, _PipelineState, _SENTINEL


def test_end_to_end_with_mock():
    """Test the full flow with mocked VLXSeek worker to avoid heavy dependencies."""
    print("Testing end-to-end multi-GPU queue mode...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        img_dir = Path(tmpdir) / "test_images"
        img_dir.mkdir()
        test_images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
        for img_name in test_images:
            (img_dir / img_name).touch()
        
        output_path = Path(tmpdir) / "output.json"
        # Ensure the output file exists to avoid FileNotFoundError
        output_path.write_text('{}', encoding='utf-8')
        
        # Setup args
        args = Mock()
        args.image_dir = str(img_dir)
        args.categories = "person;car"
        args.output = str(output_path)
        args.gpu_ids = "0,1"  # Use 2 GPUs for testing
        args.resume = False
        args.prompt_batch_size = 0
        args.queue_batch_size = 2  # Small batch size for testing
        
        # Mock everything that would require actual model loading or VLXSeek
        with patch('distill.generate_pseudo_labels.mp') as mock_mp, \
             patch('distill.generate_pseudo_labels._collect_done_names') as mock_collect, \
             patch('distill.generate_pseudo_labels._create_worker') as mock_create_worker, \
             patch('distill.generate_pseudo_labels.run_pipeline') as mock_run_pipeline, \
             patch('distill.generate_pseudo_labels._PipelineState') as mock_state_class, \
             patch('distill.generate_pseudo_labels._setup_logging'), \
             patch('distill.generate_pseudo_labels.merge_shards') as mock_merge:
            
            # Setup multiprocessing mocks
            mock_ctx = Mock()
            mock_mp.get_context.return_value = mock_ctx
            
            mock_task_queue = Mock()
            mock_done_queue = Mock()
            mock_ctx.Queue.side_effect = [mock_task_queue, mock_done_queue]
            
            mock_collect.return_value = set()  # No pre-processed images
            
            # Mock worker creation and state
            mock_worker = Mock()
            mock_create_worker.return_value = mock_worker
            mock_state = Mock()
            mock_state_class.return_value = mock_state
            
            # Mock run_pipeline to return processed image names
            def mock_run_pipeline_side_effect(*args, **kwargs):
                # Extract image_paths from kwargs or args
                image_paths = kwargs.get('image_args') or (args[1] if len(args) > 1 else [])
                # Return the same names as processed (simulating successful processing)
                return [str(p) for p in image_paths] if image_paths else []
            
            mock_run_pipeline.side_effect = mock_run_pipeline_side_effect
            
            # Mock process behavior
            mock_process = Mock()
            mock_process.is_alive.side_effect = [True, True, False, False]  # Start alive, then die
            mock_process.exitcode = 0
            mock_ctx.Process.return_value = mock_process
            
            # Setup done_queue to return acks then empty
            ack_responses = [
                ["img1.jpg", "img2.jpg"],  # First batch from GPU 0
                ["img3.jpg", "img4.jpg"],  # Second batch from GPU 1
                ["img5.jpg"],              # Third batch (remaining)
                Exception,                 # Timeout - no more work
                Exception,                 # Second timeout
            ]
            mock_done_queue.get.side_effect = ack_responses
            
            # Mock merge_shards to write dummy merged JSON
            def fake_merge(shard_outputs, output_path):
                # Write a minimal valid COCO-like structure
                data = {"images": [], "annotations": []}
                Path(output_path).write_text(json.dumps(data), encoding='utf-8')
            mock_merge.side_effect = fake_merge
            
            # Run the function
            print("Calling run_multigpu...")
            run_multigpu(args)
            
            # Verify calls
            print(f"Number of processes spawned: {mock_ctx.Process.call_count}")
            assert mock_ctx.Process.call_count == 2, f"Expected 2 processes, got {mock_ctx.Process.call_count}"
            
            # Verify queue puts (should be images + sentinels)
            print(f"Number of queue puts: {mock_task_queue.put.call_count}")
            # Should have put 5 images + 2 sentinels = 7 puts minimum
            assert mock_task_queue.put.call_count >= 7, f"Expected at least 7 queue puts, got {mock_task_queue.put.call_count}"
            
            # Verify run_pipeline was called
            print(f"Number of run_pipeline calls: {mock_run_pipeline.call_count}")
            assert mock_run_pipeline.call_count >= 3, f"Expected at least 3 run_pipeline calls, got {mock_run_pipeline.call_count}"
            
            # Verify merge_shards was called
            assert mock_merge.called, "merge_shards should have been called"
            
            print("✅ Integration test passed!")
            return True


if __name__ == "__main__":
    success = test_end_to_end_with_mock()
    if success:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)