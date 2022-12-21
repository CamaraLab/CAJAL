from pandas._libs.internals import BlockPlacement as BlockPlacement
from pandas._typing import ArrayLike as ArrayLike
from pandas.core.internals.blocks import Block as Block
from pandas.core.internals.managers import BlockManager as BlockManager
from typing import NamedTuple

class BlockPairInfo(NamedTuple):
    lvals: ArrayLike
    rvals: ArrayLike
    locs: BlockPlacement
    left_ea: bool
    right_ea: bool
    rblk: Block

def operate_blockwise(left: BlockManager, right: BlockManager, array_op) -> BlockManager: ...
def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool: ...
