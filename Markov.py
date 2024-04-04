from random import choice


# A linked list node storing a word, a reference to the next node, the word's
# occurrence count, and (for nodes not in the deepest level) a reference to
# another linked list.
#
# There will be a large number of nodes and many accesses to their attributes,
# so properties and other methods are left out to avoid function call overhead.
class Node:
    def __init__(self, data):
        self.next = None
        self.data = data
        self.cnt = 0 # Initial count of 0 allows agnostic increment in caller

# A linked list with nodes of the above type.
# 
# A count attribute is added to the list object for the highest level lists to
# track unigram frequencies without adding another level of object composition
# to the hash table's (dict's) values.
class LinkedList:
    def __init__(self):
        self.head = None
    
    # Retrieve the specified data's Node, or None if it does not exist.
    #
    # Linear search is used. Sorting and binary search are not used since
    # binary search with a linked list still takes linear time.
    def __getitem__(self, data):
        node = self.head
        while node is not None:
            if node.data == data:
                return node # Word found
            node = node.next
        
        return None # Word not found
    
    # Return an iterator over the nodes, starting from the head
    def __iter__(self):
        return LinkedListIterator(self.head)
    
    # Return an iterator over the nodes, starting right before the head.
    # 
    # This is used in conjunction with move_next_to_head().
    def lagging_iter(self):
        node_before_head = Node(None)
        node_before_head.next = self.head

        return LinkedListIterator(node_before_head)

    # Create a new node for the specified data, add the node before the head,
    # update the head reference, and return a reference to the new node.
    # 
    # Returning a reference to the new node simplifies calling code in this
    # program.
    def prepend(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node

        return node
    
    # Remove the node after the specified node from the list and reinsert it
    # before the current head. NOP if prev_node is None or before the head.
    # 
    # This handles the possible None (for empty lists) and node_before_head
    # (when the best choice is already the head) that may be produced in
    # finish.
    # 
    # This allows constant-time access of the best choice in a given list when
    # the best choice becomes the head.
    def move_next_to_head(self, prev_node):
        if prev_node is not None and prev_node.next is not self.head:
            new_head = prev_node.next
            prev_node.next = prev_node.next.next

            new_head.next = self.head
            self.head = new_head

# A forward iterator over the nodes of a linked list.
# 
# Note that the nodes (not directly the data in the nodes) are returned by
# next().
class LinkedListIterator:
    def __init__(self, head):
        self.node = head

    def __iter__(self):
        return self
    
    # Return the current node and advance the reference, or stop iteration if
    # the current node is None (end of list).
    def __next__(self):
        if self.node is None:
            raise StopIteration

        popped = self.node
        self.node = popped.next
        return popped


# A hash table of linked lists of linked lists, representing a trigram Markov
# model.
#
# To use this class:
#   1. Build a model:
#       a. Use the constructor to make a new model.
#       b. For each input file:
#           i. Call start_input(first_word, second_word).
#           ii. Call consume_word(word) for each of the following words.
#           iii. Call end_input() after the last word.
#       c. Call finish() on the object.
#   2. Generate output (may be done multiple times if the model does not change)
#       a. Call output_generator(refresh_limit) on the model.
#               This provides an output generator.
#               refresh_limit specifies the maximum number of words generated
#               before a random first word is chosen.
#       b. Call generate_word() on the output generator the desired number of times.
# 
# Note that this structure allows multiple stories to be generated easily from
# the same model.
class TrigramModel:
    def __init__(self):
        self.first_words = dict() # Hash table


    # Word counting methods

    # Count unigram and get associated list of bigrams for given first word.
    def _count_unigram(self, word):
        bigram_list = self.first_words.get(word)
        if bigram_list is None: # New unigram
            self.first_words[word] = bigram_list = LinkedList() # Add new high-level list
            bigram_list.cnt = 0 # Attach unigram counter to list
        
        bigram_list.cnt += 1 # Count instance of unigram

        return bigram_list
    
    # Count bigram and get associated list of trigrams for given second word.
    def _count_bigram(self, bigram_list, word):
        bigram_node = bigram_list[word]
        if bigram_node is None: # New bigram
            bigram_node = bigram_list.prepend(word) # Add new high-level list node
            bigram_node.child = LinkedList() # Add new low-level list
        
        bigram_node.cnt += 1 # Count instance of bigram

        return bigram_node.child

    # Count trigram for given third word.
    def _count_trigram(self, trigram_list, word):
        trigram_node = trigram_list[word]
        if trigram_node is None: # New trigram
            trigram_node = trigram_list.prepend(word) # Add new low-level list node
        
        trigram_node.cnt += 1 # Count instance of trigram


    # External training methods

    # Set first and second words to prepare for new input file
    def start_input(self, first_word, second_word):
        self.prev_prev = first_word # Second to last word
        self.prev = second_word # Last word

    # Count the last two words received as unigrams and as a single bigram.
    def end_input(self):
        self._count_bigram( # secondToLastWord lastWord
            self._count_unigram(self.prev_prev), self.prev
        )
        self._count_unigram(self.prev) # lastWord

    # Count the trigram ending in this word.
    # This involves updating the count variables for the first word as a
    # unigram, the first and second words as a bigram, and all three words as a
    # trigram.
    def consume_word(self, word):
        # Count unigram for first word, bigram for first and second word, and
        # trigram for all three words.
        bigram_list = self._count_unigram(self.prev_prev)
        trigram_list = self._count_bigram(bigram_list, self.prev)
        self._count_trigram(trigram_list, word)

        # Track last two words
        self.prev_prev = self.prev
        self.prev = word

    # Calculate the probability of each second word given each first word and
    # the probability of each third word given each first two words.
    # 
    # Rather than explicitly calculating and storing the probabilites, move the
    # most probable bigram/trigram (the one with the highest count in its list)
    # to the head of its respective list for simpler and faster output
    # generation.
    # 
    # Trigram probability: Count(first second third) / Count(first second)
    # Bigram probability: Count(first second) / Count(first)
    # Because the probabilities for bigrams/trigrams are only compared for the same
    # starting unigram/bigram (allowing the count in the numerator to provide the
    # exact same ordering as the resulting probability), calculating the
    # probabilities explicitly is not necessary, and the bigram/trigram counts are
    # used instead.
    def finish(self):
        for firstWord in self.first_words.values(): # For each first word
            # bigram_choice is the probability of the best bigram, the node
            # that appears immediately before that bigram in the list, and an
            # iterator one node behind the loop's iterator
            # 
            # The previous node is tracked to move the best bigram node to the
            # head at the end of this iteration.
            bigram_choice = [-1, None, firstWord.lagging_iter()]
            for secondWordNode in firstWord: # For each second word
                # Update best bigram choice
                _update_choice(bigram_choice, secondWordNode.cnt)

                # As for bigram_choice
                trigram_choice = [-1, None, secondWordNode.child.lagging_iter()]
                for thirdWordNode in secondWordNode.child: # For each third word
                    _update_choice(trigram_choice, thirdWordNode.cnt)

                secondWordNode.child.move_next_to_head(trigram_choice[1]) # Best trigram to head
            
            firstWord.move_next_to_head(bigram_choice[1]) # Best bigram to head
    
    # Create a new output generator based on this model
    def output_generator(self, refresh_limit):
        return OutputGenerator(self.first_words, refresh_limit)

# If a new highest-probability bigram/trigram is found, update recorded choice.
# The lagging iterator is also always advanced to ensure it stays one node
# behind the iterator of the associated loop.
def _update_choice(best_choice, new_cnt):
    if new_cnt >= best_choice[0]: # New best choice
        best_choice[0], best_choice[1] = new_cnt, next(best_choice[2])
    else: # Best choice unchanged
        next(best_choice[2])

# Based on the provided model structure, generate a sequence of words.
class OutputGenerator:
    def __init__(self, first_words, refresh_limit):
        self.first_words = first_words

        # Repetition avoidance
        self.refresh_limit = refresh_limit
        self.refresh_cnt = refresh_limit # Words until next forced random word
        self.prev_4 = None # Fourth to last word (detect 3-word cycles)
        self.prev_3 = None # Third to last word (detect 2-word cycles)

        # Bigram/trigram generation
        self.prev_prev = None # Second to last word
        self.prev = None # Last word
    
    # Choose a new word from the list of unigrams randomly.
    def _rand_word(self):
        self.refresh_cnt = self.refresh_limit # Reset refresh counter

        return choice(tuple(self.first_words.keys()))
    
    # Choose the most likely second word given the last generated word.
    def _best_bigram(self):
        # Because the last word in an input file is counted as a unigram, it is
        # possible for a unigram to have an empty list of next words (when the
        # final word in the text does not appear before the last word in any of
        # the files). In this case, a random unigram is once again chosen.
        bigram_node = self.first_words[self.prev].head

        return bigram_node.data if bigram_node is not None\
               else self._rand_word()

    # Choose the most likely third word given the last two generated words.
    def _best_trigram(self):
        # Because the last two words in an input file are counted as a bigram,
        # it is possible for a bigram to have an empty list of next words (when
        # the final bigram in an input file does not appear before the last two
        # words in any of the files). In this case, the best bigram based on
        # the last word is once again chosen; this may also result in a random
        # word being chosen if the last word also does not appear in an earlier
        # part of any input file.
        # Because this case is also addressed for choosing bigrams, it is
        # further possible for the last word generated to be random and not
        # ever appear immediately after the second to last word. Consequently,
        # there will be no bigram associated with the last two words, and the
        # best bigram for the last word will then be used (or a random word if
        # that bigram also does not exist).
        # 
        # This allows for the best trigram to be chosen whenever possible, the
        # best bigram to be chosen if no trigram is possible, and finally a
        # random word if no predictions can be made.
        bigram_node = self.first_words[self.prev_prev][self.prev]
        
        # Best trigram if prefix bigram exists and trigram exists, else best bigram
        return bigram_node.child.head.data if bigram_node is not None and\
                                              bigram_node.child.head is not None\
               else self._best_bigram() # Trigram or prefix bigram DNE

    # At the start of the generation (or when no data exists for prediction),
    # generate a random word, then the most likely bigram beginning with that
    # word, and then repeatedly the most likely trigram.
    def generate_word(self):
        new_word = self._rand_word() if self.refresh_cnt <= 0\
                   else self._best_trigram() if self.prev_prev is not None\
                   else self._best_bigram() if self.prev is not None\
                   else self._rand_word()

        # Identify 2-word and 3-word cycles and start with a new random word.
        # If a b → a, and b a → b, output loops a b a b ....
        # If a b → c, b c → a, and c a → b, output loops a b c a b c ....
        if ((new_word == self.prev_prev and self.prev == self.prev_3) or # a b a b
           (new_word == self.prev_3 and self.prev == self.prev_4)): # a b c a b
            new_word = self._rand_word()
        
        # Decrement counter until refresh
        self.refresh_cnt -= 1

        # Shift previous words window forward by 1
        self.prev_4 = self.prev_3
        self.prev_3 = self.prev_prev
        self.prev_prev = self.prev
        self.prev = new_word

        return new_word
