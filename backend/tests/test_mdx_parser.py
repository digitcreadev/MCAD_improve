
import unittest

from backend.mcad.mdx_parser import parse_mdx


class TestMdxParser(unittest.TestCase):
    def test_parse_basic_mdx(self):
        mdx = """
        SELECT {[Measures].[Margin%]} ON COLUMNS,
               {[Time].[Month].Members} ON ROWS
        FROM [Sales]
        WHERE ([Product].[Category].[Health Food], [Store].[Region].[North], [Time].[Year].[1998])
        """
        q = parse_mdx(mdx)
        self.assertEqual(q['cube'], 'Sales')
        self.assertIn('Margin%', q['measures'])
        self.assertIn('Time.Month', q['group_by'])
        self.assertEqual(q['slicers'].get('Product.Category'), 'Health Food')
        self.assertEqual(q['slicers'].get('Store.Region'), 'North')
        self.assertEqual(q['slicers'].get('Time.Year'), '1998')
        self.assertEqual(q['window_start'], '1998-01-01')
        self.assertEqual(q['window_end'], '1998-12-31')

    def test_parse_corr_and_with(self):
        mdx = """
        WITH MEMBER [Measures].[Growth] AS ([Measures].[Store Sales])
        SELECT {CORR([Measures].[Margin%],[Measures].[StockoutRate])} ON COLUMNS,
               {[Time].[Month].Members} ON ROWS
        FROM [Sales]
        WHERE ([Store].[Region].[North], [Time].[Year].[1998])
        """
        q = parse_mdx(mdx)
        self.assertIn('CORR', q['analytics'])
        self.assertTrue(q['calculated_members'])


if __name__ == '__main__':
    unittest.main()
