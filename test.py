import unittest
from app import load_data

class TestApp(unittest.TestCase):
    def test_load_data(self):
        X, Y = load_data('X.csv', 'Y.csv')
        self.assertEqual(len(X), len(Y))

if __name__ == '__main__':
    unittest.main()
