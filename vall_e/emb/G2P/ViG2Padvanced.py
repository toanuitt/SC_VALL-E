import re
import itertools
from typing import Tuple, List, Optional, Dict, Set

class VietnameseTextProcessor:
    # Constants for component types
    INITIAL = 0x001
    MEDIAL = 0x010
    FINAL = 0x100

    # Vietnamese syllable components
    CHAR_LISTS = {
        INITIAL: [
            'b', 'ch', 'd', 'đ', 'g', 'gh', 'gi', 'h', 'k', 'kh',
            'l', 'm', 'n', 'nh', 'ng', 'ngh', 'ph', 'qu', 'r', 's',
            't', 'th', 'tr', 'v', 'x'
        ],
        MEDIAL: [
            'a', 'à', 'á', 'ả', 'ã', 'ạ',
            'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
            'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',
            'e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
            'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ',
            'i', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
            'o', 'ò', 'ó', 'ỏ', 'õ', 'ọ',
            'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
            'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
            'u', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
            'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự',
            'y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',
            'ia', 'ưa', 'ua'  # Special compound vowels
        ],
        FINAL: [
            'c', 'ch', 'm', 'n', 'ng', 'nh', 'p', 't'
        ]
    }

    # Tone marks
    TONE_MARKS = {
        'ngang': '',      # Level tone (default)
        'huyền': '\u0300',  # Low falling tone (à)
        'sắc': '\u0301',    # High rising tone (á)
        'hỏi': '\u0309',    # Low rising tone (ả)
        'ngã': '\u0303',    # High broken tone (ã)
        'nặng': '\u0323'    # Low broken tone (ạ)
    }

    # Vowel to base form mapping
    VOWEL_TO_BASE = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ằ': 'ă', 'ắ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
        'ầ': 'â', 'ấ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ề': 'ê', 'ế': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ồ': 'ô', 'ố': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
        'ờ': 'ơ', 'ớ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ừ': 'ư', 'ứ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
    }

    def __init__(self):
        """Initialize the Vietnamese Text Processor."""
        self.CHAR_SETS = {k: set(v) for k, v in self.CHAR_LISTS.items()}
        self.CHARSET = set(itertools.chain(*self.CHAR_SETS.values()))
        self.CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                           for k, v in self.CHAR_LISTS.items()}

    def is_vietnamese_syllable(self, text: str) -> bool:
        """
        Check if a string is a valid Vietnamese syllable.
        
        Args:
            text: Input string to check
            
        Returns:
            bool: True if valid Vietnamese syllable, False otherwise
        """
        if not text:
            return False
            
        # Basic syllable pattern
        pattern = r'^(?:' + '|'.join(self.CHAR_LISTS[self.INITIAL]) + r')?' + \
                 r'(?:' + '|'.join(self.CHAR_LISTS[self.MEDIAL]) + r')' + \
                 r'(?:' + '|'.join(self.CHAR_LISTS[self.FINAL]) + r')?$'
        
        return bool(re.match(pattern, text.lower()))

    def get_tone(self, syllable: str) -> str:
        """
        Extract the tone from a Vietnamese syllable.
        
        Args:
            syllable: Input syllable
            
        Returns:
            str: Tone mark or empty string if no tone
        """
        for char in syllable:
            normalized = self.normalize_char(char)
            if normalized != char:
                return char[len(normalized):]
        return ''

    def normalize_char(self, char: str) -> str:
        """
        Remove tone mark from a Vietnamese character.
        
        Args:
            char: Input character
            
        Returns:
            str: Character without tone mark
        """
        return self.VOWEL_TO_BASE.get(char, char)

    def split_syllable_char(self, syllable: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Split a Vietnamese syllable into its components.
        
        Args:
            syllable: Input syllable
            
        Returns:
            Tuple of (initial consonant, main vowel, final consonant)
        """
        syllable = syllable.lower()
        
        # Find initial consonant
        initial = None
        for cons in sorted(self.CHAR_LISTS[self.INITIAL], key=len, reverse=True):
            if syllable.startswith(cons):
                initial = cons
                syllable = syllable[len(cons):]
                break
                
        # Find final consonant
        final = None
        for cons in sorted(self.CHAR_LISTS[self.FINAL], key=len, reverse=True):
            if syllable.endswith(cons):
                final = cons
                syllable = syllable[:-len(cons)]
                break
                
        # The remaining part is the medial (vowel)
        medial = syllable
        
        return initial, medial, final

    def join_syllable_char(self, initial: Optional[str], medial: str, final: Optional[str], 
                          tone: Optional[str] = None) -> str:
        """
        Join syllable components into a complete syllable.
        
        Args:
            initial: Initial consonant or None
            medial: Main vowel
            final: Final consonant or None
            tone: Tone mark or None
            
        Returns:
            str: Complete syllable
        """
        result = ''
        if initial:
            result += initial
        result += medial
        if final:
            result += final
        if tone:
            # Apply tone mark to the main vowel
            main_vowel_idx = self._find_main_vowel_index(result)
            if main_vowel_idx >= 0:
                result = result[:main_vowel_idx] + self._apply_tone(result[main_vowel_idx], tone) + result[main_vowel_idx + 1:]
        return result

    def _find_main_vowel_index(self, syllable: str) -> int:
        """Find the index of the main vowel in a syllable."""
        for i, char in enumerate(syllable):
            if char.lower() in self.VOWEL_TO_BASE or char.lower() in self.CHAR_LISTS[self.MEDIAL]:
                return i
        return -1

    def _apply_tone(self, vowel: str, tone: str) -> str:
        """Apply a tone mark to a vowel."""
        base = self.normalize_char(vowel)
        return base + tone if base else vowel

    def viet_g2p(self, text: str) -> str:
        """
        Convert Vietnamese text to phonetic representation.
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            str: Phonetic representation
        """
        words = text.split()
        result = []
        
        for word in words:
            if self.is_vietnamese_syllable(word):
                initial, medial, final = self.split_syllable_char(word)
                tone = self.get_tone(word)
                
                # Convert to IPA-like representation
                phonetic = ''
                if initial:
                    phonetic += self._convert_initial_to_ipa(initial)
                phonetic += self._convert_medial_to_ipa(medial)
                if final:
                    phonetic += self._convert_final_to_ipa(final)
                if tone:
                    phonetic += self._convert_tone_to_ipa(tone)
                    
                result.append(phonetic)
            else:
                result.append(word)
                
        return ' '.join(result)

    def _convert_initial_to_ipa(self, initial: str) -> str:
        """Convert initial consonant to IPA."""
        ipa_map = {
            'b': 'b', 'ch': 'c', 'd': 'z', 'đ': 'd', 'g': 'ɣ',
            'gh': 'ɣ', 'gi': 'z', 'h': 'h', 'k': 'k', 'kh': 'x',
            'l': 'l', 'm': 'm', 'n': 'n', 'nh': 'ɲ', 'ng': 'ŋ',
            'ngh': 'ŋ', 'ph': 'f', 'qu': 'kw', 'r': 'ʐ', 's': 's',
            't': 't', 'th': 'tʰ', 'tr': 'ʈ', 'v': 'v', 'x': 's'
        }
        return ipa_map.get(initial, initial)

    def _convert_medial_to_ipa(self, medial: str) -> str:
        """Convert medial vowel to IPA."""
        # Remove tone marks for IPA conversion
        base_medial = ''.join(self.normalize_char(c) for c in medial)
        
        ipa_map = {
            'a': 'a', 'ă': 'æ', 'â': 'ə',
            'e': 'ɛ', 'ê': 'e',
            'i': 'i',
            'o': 'ɔ', 'ô': 'o', 'ơ': 'əː',
            'u': 'u', 'ư': 'ɯ',
            'y': 'i',
            'ia': 'iə', 'ưa': 'ɯə', 'ua': 'uə'
        }
        return ipa_map.get(base_medial, base_medial)

    def _convert_final_to_ipa(self, final: str) -> str:
        """Convert final consonant to IPA."""
        ipa_map = {
            'c': 'k', 'ch': 'c', 'm': 'm', 'n': 'n',
            'ng': 'ŋ', 'nh': 'ɲ', 'p': 'p', 't': 't'
        }
        return ipa_map.get(final, final)

    def _convert_tone_to_ipa(self, tone: str) -> str:
        """Convert tone mark to IPA tone number."""
        tone_map = {
            '': '1',      # Level tone
            '\u0300': '2',  # Low falling tone
            '\u0301': '3',  # High rising tone
            '\u0309': '4',  # Low rising tone
            '\u0303': '5',  # High broken tone
            '\u0323': '6'   # Low broken tone
        }
        return tone_map.get(tone, '')

# Example usage:
if __name__ == "__main__":
    processor = VietnameseTextProcessor()
    
    # Test syllable splitting
    word = "tiếng"
    initial, medial, final = processor.split_syllable_char(word)
    print(f"Syllable components of '{word}': {initial}, {medial}, {final}")
    
    # Test syllable joining
    joined = processor.join_syllable_char('t', 'iê', 'ng', '\u0301')
    print(f"Joined syllable: {joined}")
    
    # Test G2P conversion
    text = "tiếng việt"
    phonetic = processor.viet_g2p(text)
    print(f"Phonetic representation of '{text}': {phonetic}")
