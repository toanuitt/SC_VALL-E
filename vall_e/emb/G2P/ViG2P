def VietG2P(text):
    '''
    Vietnamese Grapheme to Phoneme converter
    Converts Vietnamese text to phonetic representation
    
    Parameters:
        text (str): Input Vietnamese text
    Returns:
        str: Phonetic representation
    '''
    
    def readRules(rule_book):
        # Read pronunciation rules from file
        # Similar structure to original but with Vietnamese rules
        f = open(rule_book, 'r', encoding="utf-8")
        rule_in = []
        rule_out = []
        while True:
            line = f.readline()
            line = re.sub('\n', '', line)
            if line != '':
                if line[0] != '#':
                    IOlist = line.split('\t')
                    rule_in.append(IOlist[0])
                    rule_out.append(IOlist[1] if len(IOlist) > 1 else '')
            if not line: break
        f.close()
        return rule_in, rule_out

    # Vietnamese phoneme sets
    INITIAL = [
        't', 'th', 'đ', 'tr', 'ch', 'k', 'kh', 'g', 'gh', 
        'ng', 'ngh', 'p', 'ph', 'b', 'n', 'nh', 'm', 'l', 
        'r', 's', 'x', 'h', 'v', 'gi', 'qu'
    ]
    
    FINAL = [
        'p', 't', 'c', 'ch', 'm', 'n', 'ng', 'nh',
        'i', 'y', 'u', ''
    ]
    
    VOWELS = [
        'a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y',
        'ai', 'ao', 'au', 'ay', 'ây', 'eo', 'êu', 'ia', 'iê', 'iu',
        'oa', 'oă', 'oe', 'oi', 'ôi', 'ơi', 'ua', 'uâ', 'ue', 'ui',
        'ưa', 'ưi', 'ưu', 'uy'
    ]
    
    TONES = {
        'á': 1, 'à': 2, 'ả': 3, 'ã': 4, 'ạ': 5,
        'ấ': 1, 'ầ': 2, 'ẩ': 3, 'ẫ': 4, 'ậ': 5,
        'ắ': 1, 'ằ': 2, 'ẳ': 3, 'ẵ': 4, 'ặ': 5,
        'é': 1, 'è': 2, 'ẻ': 3, 'ẽ': 4, 'ẹ': 5,
        'ế': 1, 'ề': 2, 'ể': 3, 'ễ': 4, 'ệ': 5,
        'í': 1, 'ì': 2, 'ỉ': 3, 'ĩ': 4, 'ị': 5,
        'ó': 1, 'ò': 2, 'ỏ': 3, 'õ': 4, 'ọ': 5,
        'ố': 1, 'ồ': 2, 'ổ': 3, 'ỗ': 4, 'ộ': 5,
        'ớ': 1, 'ờ': 2, 'ở': 3, 'ỡ': 4, 'ợ': 5,
        'ú': 1, 'ù': 2, 'ủ': 3, 'ũ': 4, 'ụ': 5,
        'ứ': 1, 'ừ': 2, 'ử': 3, 'ữ': 4, 'ự': 5,
        'ý': 1, 'ỳ': 2, 'ỷ': 3, 'ỹ': 4, 'ỵ': 5
    }

    def syllable_parse(syllable):
        """Parse a Vietnamese syllable into initial, vowel, final and tone components"""
        syllable = syllable.lower()
        
        # Extract tone
        tone = 0  # Neutral tone
        for char in syllable:
            if char in TONES:
                tone = TONES[char]
                break
        
        # Remove tone marks for further processing
        for marked, unmarked in [
            ('áàảãạ', 'a'), ('ấầẩẫậ', 'â'), ('ắằẳẵặ', 'ă'),
            ('éèẻẽẹ', 'e'), ('ếềểễệ', 'ê'), ('íìỉĩị', 'i'),
            ('óòỏõọ', 'o'), ('ốồổỗộ', 'ô'), ('ớờởỡợ', 'ơ'),
            ('úùủũụ', 'u'), ('ứừửữự', 'ư'), ('ýỳỷỹỵ', 'y')
        ]:
            for c in marked:
                syllable = syllable.replace(c, unmarked)
        
        # Find initial consonant
        initial = ''
        for cons in sorted(INITIAL, key=len, reverse=True):
            if syllable.startswith(cons):
                initial = cons
                syllable = syllable[len(cons):]
                break
        
        # Find final consonant
        final = ''
        for cons in sorted(FINAL, key=len, reverse=True):
            if syllable.endswith(cons):
                final = cons
                syllable = syllable[:-len(cons)]
                break
        
        # What remains is the vowel
        vowel = syllable
        
        return initial, vowel, final, tone

    def convert_to_ipa(initial, vowel, final, tone):
        """Convert syllable components to IPA"""
        # This is a simplified IPA conversion
        ipa = ''
        
        # Convert initial
        initial_map = {
            't': 't', 'th': 'tʰ', 'đ': 'd', 'tr': 'ʈ', 'ch': 'c',
            'k': 'k', 'kh': 'kʰ', 'g': 'ɣ', 'gh': 'ɣ',
            'ng': 'ŋ', 'ngh': 'ŋ', 'p': 'p', 'ph': 'f',
            'b': 'b', 'n': 'n', 'nh': 'ɲ', 'm': 'm',
            'l': 'l', 'r': 'z', 's': 's', 'x': 's',
            'h': 'h', 'v': 'v', 'gi': 'z', 'qu': 'kw'
        }
        ipa += initial_map.get(initial, '')
        
        # Convert vowel
        vowel_map = {
            'a': 'a', 'ă': 'æ', 'â': 'ə', 'e': 'ɛ',
            'ê': 'e', 'i': 'i', 'o': 'ɔ', 'ô': 'o',
            'ơ': 'ɤ', 'u': 'u', 'ư': 'ɯ', 'y': 'i'
        }
        ipa += vowel_map.get(vowel, vowel)
        
        # Convert final
        final_map = {
            'p': 'p̚', 't': 't̚', 'c': 'k̚', 'ch': 'k̚',
            'm': 'm', 'n': 'n', 'ng': 'ŋ', 'nh': 'ɲ'
        }
        ipa += final_map.get(final, '')
        
        # Add tone mark
        tone_marks = ['', '˦', '˨', '˧˩', '˨˦', '˨˩']
        ipa += tone_marks[tone]
        
        return ipa

    def process_text(text):
        """Process full text input"""
        words = text.strip().split()
        result = []
        
        for word in words:
            syllables = word.split('-')
            word_ipa = []
            
            for syllable in syllables:
                if not syllable:
                    continue
                initial, vowel, final, tone = syllable_parse(syllable)
                ipa = convert_to_ipa(initial, vowel, final, tone)
                word_ipa.append(ipa)
            
            result.append('-'.join(word_ipa))
        
        return ' '.join(result)

    return process_text(text)
