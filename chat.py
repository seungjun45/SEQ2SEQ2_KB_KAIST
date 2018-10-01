import tensorflow as tf
import numpy as np
import math
import sys

from config import FLAGS
from model import Seq2Seq
from dialog import Dialog

import pdb


class ChatBot:

    def __init__(self, voc_path, train_dir):
        self.dialog = Dialog()
        self.dialog.load_vocab(voc_path)

        self.model = Seq2Seq(self.dialog.vocab_size)

        self.sess = tf.Session()
        #pdb.set_trace()
        ckpt = tf.train.get_checkpoint_state(train_dir)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self):
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            print(self._get_replay(line.strip()))

            sys.stdout.write("\n> ")
            sys.stdout.flush()

            line = sys.stdin.readline()

    def _decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist()

        input_len = int(math.ceil((len(enc_input) + 1) * 1.5))

        enc_input, dec_input, _ = self.dialog.transform(enc_input, dec_input,
                                                        input_len,
                                                        FLAGS.max_decode_len)

        return self.model.predict(self.sess, [enc_input], [dec_input])


###########################################################################################
    def _get_replay(self, msg): # 실제 Reply(응답) 구성하는 부분
        enc_input = self.dialog.tokenizer(msg)
        enc_input = self.dialog.tokens_to_ids(enc_input)
        dec_input = []

        curr_seq = 0
        for i in range(FLAGS.max_decode_len):
            print("Decoding 관련 내용을 넣어주세요")
            dec_input

        reply = self.dialog.decode([dec_input], True)

        return reply
###########################################################################################

def main(_):
    print("잠시만 기다려주세요...\n")

    chatbot = ChatBot(FLAGS.voc_path, FLAGS.train_dir)
    chatbot.run()


if __name__ == "__main__":
    tf.app.run()
