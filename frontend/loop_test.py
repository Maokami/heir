from heir import compile
from heir.mlir import I64, Secret

from absl.testing import absltest  # fmt: skip


class LoopTest(absltest.TestCase):

  def test_loop(self):

    @compile()
    def loopa(a: Secret[I64]):
      result1 = a
      result2 = 0
      lb = 1
      ub = 5
      for _ in range(lb, ub):
        result1 = result1 + result1
        result2 = result2 + 1
      return result1

    self.assertEqual(32, loopa(2))

  def test_loop_one_iter_arg(self):

    @compile()
    def one_iter_arg(a: Secret[I64]):
      result1 = a
      lb = 1
      ub = 5
      for _ in range(lb, ub):
        result1 = result1 + result1
      return result1

    self.assertEqual(32, one_iter_arg(2))


if __name__ == "__main__":
  absltest.main()
