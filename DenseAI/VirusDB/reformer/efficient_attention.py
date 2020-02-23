import tensorflow as tf

def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)


def sort_key_val(t1, t2, dim=-1):

    print("T1: ", t1)
    print("T2: ", t2)

    values = tf.sort(t1, axis=dim)
    t2 = tf.broadcast_to(t2, t1.shape)
    return values, tf.gather(t2, tf.argsort(t1, axis=dim), axis=dim)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))


def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tf.reshape(tensor, [-1, last_dim])
    summed_tensors = [c.sum(axis=-1) for c in tf.chunk(tensor, chunks, axis=0)]
    return tf.reshape(tf.concat(summed_tensors, axis=0), orig_size)


class ScaleNorm(tf.keras.layers.Layer):
    def __init__(self, emb, eps):
        super(ScaleNorm, self).__init__()
        w_init = tf.random_normal_initializer()
        self.g = tf.Variable(initial_value=w_init(shape=(1,), dtype='float32'), trainable=True)
        self.eps = eps

    def call(self, inputs):
        x = inputs
        n = tf.norm(inputs, axis=-1, keepdims=True).clip_by_value(min=self.eps)
        return x / n * self.g


class WithNorm(tf.keras.layers.Layer):
    def __init__(self, norm_class, emb, fn):
        super(WithNorm, self).__init__()
        self.emb = emb
        if isinstance(norm_class, ScaleNorm):
            self.norm = norm_class(emb)
        else:
            self.norm = norm_class()

        self.fn = fn

    def call(self, inputs):
        inputs = self.norm(inputs)
        return self.fn(inputs)


class Chunk(tf.keras.layers.Layer):
    def __init__(self, chunks, fn, along_axis=-1):
        super(Chunk, self).__init__()
        self.axis = along_axis
        self.chunks = chunks
        self.fn = fn

    def call(self, inputs):
        chunks = tf.split(inputs, self.chunks, axis=self.axis)
        return tf.concat([self.fn(c) for c in chunks], axis=self.axis)


def cache_fn(f):
    cache = None

    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class LSHAttention(tf.keras.Model):
    def __init__(self,
                 dropout=0.,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 random_rotations_per_head=False):

        super(LSHAttention, self).__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dropout_for_hash = tf.keras.layers.Dropout(dropout)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

    def hash_vectors(self, n_buckets, vecs):

        batch_size = vecs.shape[0]
        # device = vecs.device

        print("batch_size: ", batch_size)
        print("n_buckets: ", n_buckets)

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape.as_list()[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        print("rotations_shape: ", rotations_shape)
        random_rotations = tf.random.normal(rotations_shape)
        random_rotations = tf.broadcast_to(random_rotations, rotations_shape)
        # random_rotations = tf.expand_dims(random_rotations, axis=-1)
        # random_rotations = tf.expand_dims(random_rotations, axis=-1)
        # random_rotations = tf.expand_dims(random_rotations, axis=-1)
        #, (
            # batch_size, vecs.shape[-1], self.n_hashes if self._rehash_each_round else 1, rot_size // 2)
        print("random_rotations： ", random_rotations)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = tf.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        print("vecs: ", vecs)
        print("rotated_vecs: ", rotated_vecs)

        if self._rehash_each_round:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            buckets = tf.math.argmax(rotated_vecs, axis=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = tf.range(self.n_hashes)
            offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))
            offsets = tf.cast(offsets, tf.int64)
            buckets = tf.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = tf.squeeze(rotated_vecs, axis=0)
            bucket_range = tf.range(rotated_vecs.shape[-1])
            bucket_range = tf.reshape(bucket_range, (1, -1))
            bucket_range = tf.broadcast_to(bucket_range, rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, axis=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape
            buckets = tf.reshape(buckets.permute((*_, h)), (-1,))

        print("buckets: ", buckets)
        return buckets

    def call(self, qk, v):

        print("QK: ", qk.shape)
        batch_size, seqlen, dim = qk.shape.as_list()
        # device = qk.device

        n_buckets = seqlen // self.bucket_size

        n_bins = n_buckets

        buckets = self.hash_vectors(n_buckets, qk)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        print("buckets:", buckets)

        ticker = tf.expand_dims(tf.range(self.n_hashes * seqlen), axis=0)
        buckets_and_t = seqlen * buckets + tf.cast((ticker % seqlen), tf.int64)
        buckets_and_t = tf.stop_gradient(buckets_and_t)

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = tf.stop_gradient(sbuckets_and_t)
        sticker = tf.stop_gradient(sticker)
        undo_sort = tf.stop_gradient(undo_sort)

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = tf.reshape(st, (batch_size, self.n_hashes * n_bins, -1))
        bqk = tf.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, sqk.shape[-1]))
        bv = tf.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, sv.shape[-1]))
        bq_buckets = bkv_buckets = tf.reshape(sbuckets_and_t // seqlen, (batch_size, self.n_hashes * n_bins, -1))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = make_unit_length(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        bkv_buckets = look_one_back(bkv_buckets)

        print("bq.shape: ", bq.shape[-1].value, type(bq.shape[-1].value))
        # Dot-product attention.

        dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (bq.shape[-1].value ** -0.5)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(mask, tf.float32)) + (1 - tf.cast(mask, tf.float32)) * float('-inf')
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots = tf.math.multiply(dots, tf.cast(self_mask, tf.float32)) + (1 - tf.cast(self_mask, tf.float32)) * (- 1e5)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(bucket_mask, tf.float32)) + (
                    1 - tf.cast(bucket_mask, tf.float32)) * float('-inf')
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % (self.n_hashes * n_bins)
            if not self._attend_across_buckets:
                locs1 = buckets * (self.n_hashes * n_bins) + locs1
                locs2 = buckets * (self.n_hashes * n_bins) + locs2
            locs = tf.transpose(
                tf.concat([
                    tf.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                    tf.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
                ], 1),
                perm=[0, 2, 1])

            slocs = batched_index_select(locs, st)
            b_locs = tf.reshape(slocs, (batch_size, self.n_hashes * n_bins, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = tf.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = tf.stop_gradient(dup_counts)
            assert dup_counts.shape == dots.shape
            dots = dots - tf.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
        dots = tf.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = tf.einsum('buij,buje->buie', dots, bv)
        so = tf.reshape(bo, (batch_size, -1, bo.shape[-1]))
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(tf.keras.layers.Layer):
            def __init__(self):
                super(UnsortLogits, self).__init__()

            def call(self, so, slogits):
                so, slogits = tf.stop_gradient(so), tf.stop_gradient(slogits)
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

        unsortlogits = UnsortLogits()
        o, logits = unsortlogits(so, slogits)

        if self.n_hashes == 1:
            out = o
        else:
            o = tf.reshape(o, (batch_size, self.n_hashes, seqlen, o.shape[-1]))
            logits = tf.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))
            probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
            out = tf.reduce_sum(o * probs, axis=1)

        assert out.shape == v.shape
        return out, buckets


class LSHSelfAttention(tf.keras.Model):
    def __init__(self,
                 embedding_size,
                 heads=8,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 attn_chunks=None,
                 random_rotations_per_head=False,
                 attend_across_buckets=True,
                 allow_duplicate_attention=True, **kwargs):
        super(LSHSelfAttention, self).__init__()
        assert embedding_size % heads == 0, 'dimensions must be divisible by number of heads'

        self.embedding_size = embedding_size
        self.heads = heads
        self.attn_chunks = heads if attn_chunks is None else attn_chunks

        self.toqk = tf.keras.layers.Dense(embedding_size, use_bias=False)
        self.tov = tf.keras.layers.Dense(embedding_size, use_bias=False)
        self.to_out = tf.keras.layers.Dense(embedding_size)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size,
                                     causal=causal,
                                     random_rotations_per_head=random_rotations_per_head,
                                     attend_across_buckets=attend_across_buckets,
                                     allow_duplicate_attention=allow_duplicate_attention,
                                     **kwargs)

    def call(self, inputs):
        batch_size, token, embedding, heads = *inputs.shape, self.heads
        assert token % self.bucket_size == 0, f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}'

        # Query and Key
        qk = self.toqk(inputs)
        print("QK: ", qk)

        # Value
        v = self.tov(inputs)
        print("V: ", v)

        def merge_heads(v):
            return tf.reshape(tf.transpose(tf.reshape(v, (batch_size, token, heads, -1)), perm=[0, 2, 1, 3]), (batch_size * heads, token, -1))

        def split_heads(v):
            return tf.transpose(tf.reshape(v, (batch_size, token, heads, -1)), perm=[0, 2, 1, 3])

        print("QK: ", qk)
        print("V: ", v)


        qk = merge_heads(qk)
        v = merge_heads(v)

        outputs = process_inputs_chunk(self.lsh_attn, qk, v, chunks=self.attn_chunks)
        # attn_out = tf.concat([output for (output, _) in outputs], axis=0)
        #
        # out = tf.reshape(split_heads(attn_out), (b, t, e))

        return outputs#self.to_out(out)


if __name__ == '__main__':
    x = tf.ones((32, 128, 128))
    print("X: ", x)

    lsh_attention_layer = LSHSelfAttention(800)
    y, v = lsh_attention_layer(x)
    print(y)