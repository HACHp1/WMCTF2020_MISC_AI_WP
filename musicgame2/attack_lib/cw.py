import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.1 # 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess


class CarliniL2:
    def __init__(
            self, model,sample_shape, batch_size=1, confidence=CONFIDENCE, targeted=TARGETED, learning_rate=LEARNING_RATE,
            binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
            initial_const=INITIAL_CONST, boxmin=-0.5, boxmax=0.5):

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        self.shape = sample_shape
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.model = model

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', imgs.shape[0])
        for i in range(0, imgs.shape[0], self.batch_size):
            print('tick', i)
            # print("imgs[i:i + self.batch_size]", imgs[i:i + self.batch_size])
            # print("targets", targets)
            r.extend(self.attack_batch(
                imgs[i:i + self.batch_size], targets))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        # print("imgs, labs in attack_batch", imgs, labs) #shape=(1, 28, 28, 1), dtype=float32) [array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])]

        batch_size = self.batch_size

        def compare(x, y):

            if not isinstance(x, (float, int, np.int64)):
                x = x.numpy()
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # @tf.function
        def train_step(modifier, timg, tlab, const):
            with tf.GradientTape() as tape:
                newimg = tf.tanh(modifier + timg) * self.boxmul + self.boxplus
                # newimg = np.random.rand(1, 28, 28, 1)
                output = self.model(newimg)
                output = tf.cast(output, dtype=tf.float32)
                
                l2dist = tf.reduce_sum(
                    tf.square(newimg - (tf.tanh(timg) * self.boxmul + self.boxplus)), range(len(self.shape)-1))

                real = tf.math.reduce_sum((tlab) * output, 1)
                other = tf.math.reduce_max(
                    (1 - tlab) * output - (tlab * 10000), 1)
                if self.TARGETED:
                    # if targetted, optimize for making the other class most likely
                    loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
                else:
                    # if untargeted, optimize for making this class least likely.
                    loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

                loss2 = tf.reduce_sum(l2dist)
                loss1 = tf.reduce_sum(const * loss1)

                loss = loss1 + loss2
            optimizer = optimizers.Adam(self.LEARNING_RATE)
            loss_metric = tf.keras.metrics.Mean(name='train_loss')

            # print(loss.shape)
            # print(loss)

            grads = tape.gradient(loss, [modifier])
            optimizer.apply_gradients(zip(grads, [modifier]))
            loss_metric.update_state(loss)
            return loss, l2dist, output, newimg, loss1, loss2

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        # print(np.shape(imgs))
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        print(np.shape(o_bestattack), "np.shape(o_bestattack)")  # (1, 28, 28, 1)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            batch = tf.Variable(imgs[:batch_size], dtype=tf.float32)
            batchlab = tf.Variable(labs[:batch_size], dtype=tf.float32)
            # print("*******batchlab***********", batchlab)  # shape=(1, 10)
            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            modifier = tf.Variable(np.zeros(self.shape, dtype=np.float32))
            const = tf.Variable(CONST, dtype=tf.float32)
            prev = np.inf
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack

                l, l2s, scores, nimg, loss1, loss2 = train_step(
                    modifier, batch, batchlab, const)

                if np.all(scores >= -.0001) and np.all(scores <= 1.0001):
                    # 检测是不是用的softmax前的一层（若和与1很接近，则很有可能是经过sm处理过）
                    if np.allclose(np.sum(scores, axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")

                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, l, loss1, loss2)
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l
                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    # print("batchlab", np.argmax(batchlab[e]))
                    # print("(sc, np.argmax(batchlab))", sc, np.argmax(sc))
                    # print("l2 and bestl2[e]", l2, bestl2[e])
                    # print("compare(sc, tf.argmax(batchlab))",
                    #       compare(sc, tf.argmax(batchlab[e])))
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

                # adjust the constant as needed
            for e in range(batch_size):
                print("bestscore[e]", bestscore[e])
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
