from googletransn import Translator

translator = Translator()
translation = translator.translate('안녕하세요.')

print('-' * 30)
print('>>> src: {}'.format(translation.src))
print('>>> dest: {}'.format(translation.dest))
print('>>> origin text: {}'.format(translation.origin))
print('>>> traslated text: {}'.format(translation.text))
print('>>> pronunciation: {}'.format(translation.pronunciation))