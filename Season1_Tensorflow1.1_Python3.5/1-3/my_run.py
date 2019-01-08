from __future__ import print_function, division
import tensorflow as tf


def basic_operation():
    v1 = tf.Variable(10, dtype=tf.int64)
    v2 = tf.Variable(5, dtype=tf.int64)

    addv = v1 + v2

    print('[v1]', v1)
    print('[addv]', addv)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    print('Variable需要初始化')
    print('[v1]', v1.eval(session=sess))
    print('[v1 + v2]', sess.run(addv))
    print('[v1 + v2]', addv.eval(session=sess))

    print('*' * 10)
    c1 = tf.constant(10)
    c2 = tf.constant(4)
    addc = c1 * c2
    print('[c1]', c1)
    print('[addc]', addc)

    print('[c1]', c1.eval(session=sess))
    print('[c1 * c2]', sess.run(addc))
    print('[c1 * c2]', addc.eval(session=sess))

    print('# 上面这种定义操作，再执行操作的模式被称之为“符号式编程” Symbolic Programming')

    graph = tf.Graph()
    with graph.as_default():
        p1 = tf.placeholder(dtype=tf.float64)
        v3 = tf.Variable([1, 4, 3, 4, 7], dtype=tf.float64)
        div = p1 / v3

    print('[p1]', p1)
    print('[v3]', v3)
    print('[div]', div)

    print('在graph中做计算')
    with tf.Session(graph=graph) as my_sess:
        tf.global_variables_initializer().run(session=my_sess)
        print('[v3]', v3.eval(session=my_sess))
        values = load_from_remote()
        for partial_val in load_partial(values, 5):
            print('[p1]', partial_val)
            # print('[div]', my_sess.run(div, feed_dict={p1: partial_val}))
            print('[div]', div.eval(session=my_sess, feed_dict={p1: partial_val}))

def load_from_remote():
    return [-x for x in range(1000)]


def load_partial(values, step):
    index = 0
    while index < len(values):
        yield values[index: index + step]
        index += step


if __name__ == '__main__':
    basic_operation()
