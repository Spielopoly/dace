# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the StorageType.CuTile_Tile enum value and its integration
with the DaCe type system mappings."""
import pytest
import dace
from dace import dtypes


class TestStorageDefaultSchedule:
    """Tests that STORAGEDEFAULT_SCHEDULE maps CuTile_Tile to CuTile schedule."""

    def test_cutile_tile_maps_to_cutile_schedule(self):
        """CuTile_Tile storage should prefer CuTile schedule."""
        assert dtypes.STORAGEDEFAULT_SCHEDULE[dtypes.StorageType.CuTile_Tile] == dtypes.ScheduleType.CuTile


class TestCanAccess:
    """Tests for can_access() with CuTile_Tile storage and CuTile schedule."""

    def test_cutile_schedule_can_access_cutile_tile(self):
        """CuTile schedule should be able to access CuTile_Tile storage."""
        assert dtypes.can_access(dtypes.ScheduleType.CuTile, dtypes.StorageType.CuTile_Tile) is True

    def test_cutile_schedule_can_access_gpu_global(self):
        """CuTile schedule should be able to access GPU_Global storage."""
        assert dtypes.can_access(dtypes.ScheduleType.CuTile, dtypes.StorageType.GPU_Global) is True

    def test_cutile_schedule_can_access_cpu_pinned(self):
        """CuTile schedule should be able to access CPU_Pinned storage."""
        assert dtypes.can_access(dtypes.ScheduleType.CuTile, dtypes.StorageType.CPU_Pinned) is True

    def test_cutile_schedule_cannot_access_cpu_heap(self):
        """CuTile schedule should NOT be able to access CPU_Heap storage."""
        assert dtypes.can_access(dtypes.ScheduleType.CuTile, dtypes.StorageType.CPU_Heap) is not True

    def test_cutile_schedule_can_access_register(self):
        """CuTile schedule should be able to access Register storage (registers are universal)."""
        assert dtypes.can_access(dtypes.ScheduleType.CuTile, dtypes.StorageType.Register) is True

    def test_gpu_device_cannot_access_cutile_tile(self):
        """GPU_Device schedule should NOT be able to access CuTile_Tile storage."""
        result = dtypes.can_access(dtypes.ScheduleType.GPU_Device, dtypes.StorageType.CuTile_Tile)
        assert result is not True

    def test_cpu_multicore_cannot_access_cutile_tile(self):
        """CPU_Multicore schedule should NOT be able to access CuTile_Tile storage."""
        result = dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, dtypes.StorageType.CuTile_Tile)
        assert result is not True


class TestCanAllocate:
    """Tests for can_allocate() with CuTile_Tile storage."""

    def test_cutile_tile_can_allocate_in_cutile(self):
        """CuTile_Tile storage should be allocatable in CuTile schedule."""
        assert dtypes.can_allocate(dtypes.StorageType.CuTile_Tile, dtypes.ScheduleType.CuTile) is True

    def test_cutile_tile_cannot_allocate_in_cpu(self):
        """CuTile_Tile storage should NOT be allocatable in CPU_Multicore schedule."""
        assert dtypes.can_allocate(dtypes.StorageType.CuTile_Tile, dtypes.ScheduleType.CPU_Multicore) is False

    def test_cutile_tile_cannot_allocate_in_gpu_device(self):
        """CuTile_Tile storage should NOT be allocatable in GPU_Device schedule."""
        assert dtypes.can_allocate(dtypes.StorageType.CuTile_Tile, dtypes.ScheduleType.GPU_Device) is False

    def test_cutile_tile_cannot_allocate_in_sequential(self):
        """CuTile_Tile storage should NOT be allocatable in Sequential schedule."""
        assert dtypes.can_allocate(dtypes.StorageType.CuTile_Tile, dtypes.ScheduleType.Sequential) is False

    def test_gpu_shared_allocation_unchanged(self):
        """GPU_Shared allocation rules should be unchanged."""
        assert dtypes.can_allocate(dtypes.StorageType.GPU_Shared, dtypes.ScheduleType.GPU_Device) is True
        assert dtypes.can_allocate(dtypes.StorageType.GPU_Shared, dtypes.ScheduleType.CPU_Multicore) is False


class TestSdfgArrayIntegration:
    """Integration tests that use CuTile_Tile storage in SDFG array definitions."""

    def test_add_array_with_cutile_tile_storage(self):
        """Should be able to create an SDFG array with CuTile_Tile storage."""
        sdfg = dace.SDFG('test_cutile_tile_array')
        sdfg.add_array('A', [64, 64], dace.float32, storage=dtypes.StorageType.CuTile_Tile)
        assert sdfg.arrays['A'].storage == dtypes.StorageType.CuTile_Tile

    def test_add_transient_with_cutile_tile_storage(self):
        """Should be able to create a transient array with CuTile_Tile storage."""
        sdfg = dace.SDFG('test_cutile_tile_transient')
        sdfg.add_transient('tmp', [32], dace.float32, storage=dtypes.StorageType.CuTile_Tile)
        assert sdfg.arrays['tmp'].storage == dtypes.StorageType.CuTile_Tile
        assert sdfg.arrays['tmp'].transient is True

    def test_sdfg_serialize_deserialize_cutile_tile(self):
        """CuTile_Tile storage should survive SDFG serialization/deserialization."""
        sdfg = dace.SDFG('test_cutile_tile_serialize')
        sdfg.add_transient('tmp', [16], dace.float32, storage=dtypes.StorageType.CuTile_Tile)

        # Serialize to JSON and back
        json_str = sdfg.to_json()
        sdfg2 = dace.SDFG.from_json(json_str)
        assert sdfg2.arrays['tmp'].storage == dtypes.StorageType.CuTile_Tile
