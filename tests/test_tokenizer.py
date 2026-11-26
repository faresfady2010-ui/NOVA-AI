"""NOVA tokenizer tests."""
import unittest
from model.tokenizer import ByteTokenizer

class TokenizerTests(unittest.TestCase):
    def test_round_trip_english_arabic_unicode(self):
        tokenizer=ByteTokenizer(); text="NOVA — مرحبًا! 🌙"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)),text)
    def test_special_tokens(self):
        tokenizer=ByteTokenizer(); ids=tokenizer.encode("abc",add_bos=True,add_eos=True)
        self.assertEqual(ids[0],tokenizer.BOS); self.assertEqual(ids[-1],tokenizer.EOS)
        self.assertEqual(tokenizer.decode(ids),"abc")
    def test_incomplete_utf8_replaced(self):
        self.assertIn("�",ByteTokenizer().decode([0xD9]))
    def test_bad_ids_rejected(self):
        with self.assertRaises(ValueError): ByteTokenizer().decode([999])

if __name__=="__main__": unittest.main()
