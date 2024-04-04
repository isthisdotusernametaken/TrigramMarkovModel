from Tester import _tokenize
from Markov import TrigramModel

words = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed eu aliquam turpis. Donec sodales semper arcu in sodales.
Integer sit amet dapibus felis. Proin metus justo, molestie a dignissim et, ultricies vel libero.
Maecenas venenatis elit at tempus venenatis. Vivamus felis arcu, dictum et leo vel, tincidunt vehicula tortor.
Praesent aliquet lobortis ante, et porttitor ante porttitor id. Cras faucibus iaculis diam, eget fringilla purus pulvinar in.

Praesent placerat, nibh id rhoncus pulvinar, nulla magna posuere augue, ac hendrerit libero mi nec sem.
Nulla non efficitur mauris, a hendrerit urna. Duis dignissim dignissim neque quis molestie.
Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae;
Nam sollicitudin tortor sit amet enim molestie fermentum. Morbi non leo odio. Quisque sit amet arcu feugiat,
pharetra metus vel, rhoncus massa. Sed in nisl in risus facilisis tincidunt et at sem. Ut facilisis viverra nibh at fringilla.
Vestibulum vitae lorem sodales, congue mi a, cursus arcu. In ex nulla, ultricies vel ligula in, luctus egestas lectus
"""

# print(tokenize(words))
tokenized = _tokenize(words)
model = TrigramModel()
model.start_input(tokenized[0], tokenized[1])
for i in range(2, len(tokenized)):
    model.consume_word(tokenized[i])
model.end_input()
model.finish()

output_gen = model.output_generator()
for i in range(100):
    print(output_gen.generate_word(), end=" ")