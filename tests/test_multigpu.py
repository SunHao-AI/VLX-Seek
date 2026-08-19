"""Tests for multi-GPU pseudo-label generation functionality."""
import multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill.generate_pseudo_labels import (
    _worker_shard,
    run_multigpu,
    _PipelineState,
    _SENTINEL,
)


def test_worker_shard_basic():
    """Test that _worker_shard processes items from queue and sends acks."""
    # Setup
    args = Mock()
    args.output = "dummy.json"
    args.device = "cpu"
    args.log_timing = False
    args.queue_batch_size = 2
    args.prompt_batch_size = 0
    args.crop_inference = False
    args.max_proposals = 100
    args.letterbox_size = 1024
    args.slice_width = 1000
    args.slice_height = 1000
    args.overlap_width_ratio = 0.1
    args.overlap_height_ratio = 0.1
    args.lang = "en"
    args.max_new_tokens = 1024
    args.temperature = 0.0
    args.detector_checkpoint = "dummy"
    args.model_path = "dummy"
    args.backend = "hf"
    args.min_area = 0.0
    
    # Create queues
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    done_queue = ctx.Queue()
    
    # Add test items
    task_queue.put(Path("img1.jpg"))
    task_queue.put(Path("img2.jpg"))
    task_queue.put(_SENTINEL)
    
    # Mock worker and run_pipeline
    with patch('distill.generate_pseudo_labels._create_worker') as mock_create_worker, \
         patch('distill.generate_pseudo_labels.run_pipeline') as mock_run_pipeline, \
         patch('distill.generate_pseudo_labels._PipelineState') as mock_state_class, \
         patch('distill.generate_pseudo_labels._setup_logging'):
        
        mock_worker = Mock()
        mock_create_worker.return_value = mock_worker
        mock_state = Mock()
        mock_state_class.return_value = mock_state
        
        # Run worker
        _worker_shard(args, 0, task_queue, done_queue, "dummy.json")
        
        # Verify
        assert mock_run_pipeline.call_count == 1  # One batch of 2 items
        # Check that ack was sent
        ack = done_queue.get(timeout=1)
        assert ack == ["img1.jpg", "img2.jpg"]


def test_worker_shard_single_batch():
    """Test worker with batch_size=1 processes items one by one."""
    args = Mock()
    args.output = "dummy.json"
    args.device = "cpu"
    args.log_timing = False
    args.queue_batch_size = 1  # Process one by one
    args.prompt_batch_size = 0
    args.crop_inference = False
    args.max_proposals = 100
    args.letterbox_size = 1024
    args.slice_width = 1000
    args.slice_height = 1000
    args.overlap_width_ratio = 0.1
    args.overlap_height_ratio = 0.1
    args.lang = "en"
    args.max_new_tokens = 1024
    args.temperature = 0.0
    args.detector_checkpoint = "dummy"
    args.model_path = "dummy"
    args.backend = "hf"
    args.min_area = 0.0
    
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    done_queue = ctx.Queue()
    
    task_queue.put(Path("img1.jpg"))
    task_queue.put(Path("img2.jpg"))
    task_queue.put(_SENTINEL)
    
    with patch('distill.generate_pseudo_labels._create_worker') as mock_create_worker, \
         patch('distill.generate_pseudo_labels.run_pipeline') as mock_run_pipeline, \
         patch('distill.generate_pseudo_labels._PipelineState') as mock_state_class, \
         patch('distill.generate_pseudo_labels._setup_logging'):
        
        mock_worker = Mock()
        mock_create_worker.return_value = mock_worker
        mock_state = Mock()
        mock_state_class.return_value = mock_state
        
        _worker_shard(args, 0, task_queue, done_queue, "dummy.json")
        
        # Should have called run_pipeline twice (once per image)
        assert mock_run_pipeline.call_count == 2
        
        # Should have sent two acks
        ack1 = done_queue.get(timeout=1)
        ack2 = done_queue.get(timeout=1)
        assert set(ack1) == {"img1.jpg"}
        assert set(ack2) == {"img2.jpg"}


def test_worker_shard_flush_on_empty_queue():
    """Test that worker flushes remaining batch when queue is empty."""
    args = Mock()
    args.output = "dummy.json"
    args.device = "cpu"
    args.log_timing = False
    args.queue_batch_size = 3  # Batch size 3
    args.prompt_batch_size = 0
    args.crop_inference = False
    args.max_proposals = 100
    args.letterbox_size = 1024
    args.slice_width = 1000
    args.slice_height = 1000
    args.overlap_width_ratio = 0.1
    args.overlap_height_ratio = 0.1
    args.lang = "en"
    args.max_new_tokens = 1024
    args.temperature = 0.0
    args.detector_checkpoint = "dummy"
    args.model_path = "dummy"
    args.backend = "hf"
    args.min_area = 0.0
    
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    done_queue = ctx.Queue()
    
    # Add only 2 items (less than batch size)
    task_queue.put(Path("img1.jpg"))
    task_queue.put(Path("img2.jpg"))
    task_queue.put(_SENTINEL)
    
    with patch('distill.generate_pseudo_labels._create_worker') as mock_create_worker, \
         patch('distill.generate_pseudo_labels.run_pipeline') as mock_run_pipeline, \
         patch('distill.generate_pseudo_labels._PipelineState') as mock_state_class, \
         patch('distill.generate_pseudo_labels._setup_logging'):
        
        mock_worker = Mock()
        mock_create_worker.return_value = mock_worker
        mock_state = Mock()
        mock_state_class.return_value = mock_state
        
        _worker_shard(args, 0, task_queue, done_queue, "dummy.json")
        
        # Should have called run_pipeline once (flushed the partial batch)
        assert mock_run_pipeline.call_count == 1
        
        # Should have sent one ack with both items
        ack = done_queue.get(timeout=1)
        assert set(ack) == {"img1.jpg", "img2.jpg"}


def test_run_multigpu_basic():
    """Test run_multigpu coordinates workers correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        # Create dummy images
        (img_dir / "img1.jpg").touch()
        (img_dir / "img2.jpg").touch()
        (img_dir / "img3.jpg").touch()

        output_path = Path(tmpdir) / "output.json"

        args = Mock()
        args.image_dir = str(img_dir)
        args.categories = "person;car"
        args.output = str(output_path)
        args.gpu_ids = "0,1"  # 2 GPUs
        args.resume = False
        args.prompt_batch_size = 0
        args.queue_batch_size = 2

        # Mock the multiprocessing parts but let file operations happen for real
        with patch('distill.generate_pseudo_labels.mp') as mock_mp:
            # Define the Empty exception that the code expects
            mock_mp.queues.Empty = Exception

            mock_ctx = Mock()
            mock_mp.get_context.return_value = mock_ctx

            mock_task_queue = Mock()
            mock_done_queue = Mock()
            mock_ctx.Queue.side_effect = [mock_task_queue, mock_done_queue]

            mock_collect = Mock()
            mock_collect.return_value = set()  # No pre-existing images

            # Mock worker and state
            mock_worker = Mock()
            mock_create_worker = Mock()
            mock_create_worker.return_value = mock_worker
            mock_state_class = Mock()
            mock_state_class.return_value = Mock()
            
            # Make run_pipeline do nothing but return success
            mock_run_pipeline = Mock()
            mock_run_pipeline.return_value = ["img1.jpg", "img2.jpg"]  # dummy return

            # Mock process
            mock_process = Mock()
            mock_process.is_alive.return_value = True
            mock_process.exitcode = 0
            mock_ctx.Process.return_value = mock_process

            # Make done_queue return acks then empty
            mock_done_queue.get.side_effect = [
                ["img1.jpg", "img2.jpg"],  # First ack
                ["img3.jpg"],              # Second ack
                mock_mp.queues.Empty(),    # Timeout to break loop
                mock_mp.queues.Empty(),    # Second timeout
            ]

            # Run
            run_multigpu(args)

            # Verify
            assert mock_ctx.Process.call_count == 2  # Two GPUs
            assert mock_task_queue.put.call_count >= 3  # At least 3 images + sentinels


if __name__ == "__main__":
    test_worker_shard_basic()
    test_worker_shard_single_batch()
    test_worker_shard_flush_on_empty_queue()
    test_run_multigpu_basic()
    print("All tests passed!")