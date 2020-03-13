import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    A = tf.reduce_sum(tf.multiply(inputs, true_w), axis=1)
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, tf.transpose(true_w))), axis =1))

    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    input_shape = inputs.get_shape().as_list()
    batch_size = input_shape[0]
    embedding_size = input_shape[1]
    sample_size = len(sample)

    unigram_prob = tf.convert_to_tensor(unigram_prob,dtype=tf.float32)
    sample = tf.convert_to_tensor(sample, dtype=tf.int32)

    # Small value to add before taking log of [batch_size,1]
    log_add_bs = [0.000000000001 for i in range(batch_size)]
    log_add_bs = tf.reshape(tf.convert_to_tensor(log_add_bs, dtype=tf.float32), [batch_size,1])

    # Small value to add before taking log of [sample_size,1]
    log_add_ss = [0.000000000001 for j in range(sample_size)]
    log_add_ss = tf.reshape(tf.convert_to_tensor(log_add_ss, dtype=tf.float32), [1, sample_size])

    # Small value to add before taking log of 2nd term [batch_size,sample_size]
    log_add_bs_ss = [[0.000000000001 for j in range(sample_size)] for i in range(batch_size)]
    log_add_bs_ss = tf.reshape(tf.convert_to_tensor(log_add_bs_ss, dtype=tf.float32), [batch_size, sample_size])


    #First term
    #===========================================================================

    # Lookup embedding of target words w_o for labels
    u_o = tf.reshape(tf.nn.embedding_lookup(weights, labels), [batch_size, embedding_size])

    # Lookup biases for target words w_o
    b_o = tf.reshape(tf.nn.embedding_lookup(biases, labels), [batch_size, 1])

    # Calculate sample probabilities from u_c.u_o + b_o
    sample_prob = tf.reshape(tf.reduce_sum(tf.multiply(inputs, u_o), axis=1), [batch_size, 1])
    sample_prob = tf.add(sample_prob, b_o)

    # Lookup labels unigram probabilities
    target_unigram_prob = tf.reshape(tf.nn.embedding_lookup(unigram_prob, labels), [batch_size, 1])

    #Calculate log(k.P(wo))
    target_unigram_prob = tf.scalar_mul(sample_size, target_unigram_prob)
    target_unigram_prob = tf.add(target_unigram_prob, log_add_bs)
    target_unigram_prob = tf.log(target_unigram_prob)

    # Calculate Pr(d=1,w_o|w_c) = sigmoid(sample_prob - log(k.P(wo)))
    P_wo = tf.reshape(tf.sigmoid(tf.subtract(sample_prob, target_unigram_prob)), [batch_size, 1])
    P_wo = tf.log(tf.add(P_wo, log_add_bs))

    #===========================================================================

    #Second term

    # Lookup embedding of negative w_x words for sample
    u_x = tf.reshape(tf.nn.embedding_lookup(weights, sample), [sample_size, embedding_size])

    # Lookup biases for negative w_x words
    b_x = tf.reshape(tf.nn.embedding_lookup(biases, sample), [sample_size, 1])
    #Transpose b_x and make it to batch_size * sample_size
    b_x = tf.transpose(b_x)
    # b_x = tf.broadcast_to(b_x, [batch_size, sample_size]) # [batch_size,sample_size]

    # Calculate s(w_x,w_c) = uc.u_x + bx
    neg_sample_prob = tf.matmul(inputs, u_x, transpose_b=True)
    neg_sample_prob = tf.reshape(tf.add(neg_sample_prob, b_x), [batch_size, sample_size])

    # Lookup negative words unigram probabilities (1,sample_size)
    neg_unigram_prob = tf.transpose(tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample), [sample_size, 1])) #[1, sample_size]

    # Calculate log(k.P(wx))
    neg_unigram_prob = tf.scalar_mul(sample_size, neg_unigram_prob)
    neg_unigram_prob = tf.add(neg_unigram_prob, log_add_ss)
    neg_unigram_prob = tf.log(neg_unigram_prob)
    # neg_unigram_prob = tf.broadcast_to(neg_unigram_prob, [batch_size, sample_size]) # [batch_size,sample_size]
    print("neg_unigram_prob: ", neg_unigram_prob)

    #Matrix of ones
    ones = [[1.0 for j in range(sample_size)] for i in range(batch_size)]
    ones = tf.reshape(tf.convert_to_tensor(ones,dtype=tf.float32), [batch_size, sample_size])
    print("ones: ", ones)

    # Calculate P_wx = sigmoid(neg_sample_prob-log(kPx))
    P_wx = tf.reshape(tf.sigmoid(tf.subtract(neg_sample_prob, neg_unigram_prob)), [batch_size, sample_size])
    print("P_wx: ", P_wx)

    # Calculate log(1- Pr(d=1,w_x|w_c))
    right_sum = tf.subtract(ones, P_wx)
    right_sum = tf.add(right_sum, log_add_bs_ss)
    right_sum = tf.log(right_sum)

    right_sum = tf.reshape(tf.reduce_sum(right_sum, axis=1), [batch_size, 1])

    nce_loss = tf.scalar_mul(-1.0, tf.add(P_wo, right_sum))

    return nce_loss