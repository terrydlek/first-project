#-*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import os
from jamo import h2j
from itertools import chain


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char


def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # load conversion dictionaries
    j2hcj, j2sj, j2shcj = load_j2hcj(), load_j2sj(), load_j2shcj()

    # Parse
    fpaths, text_lengths, texts = [], [], []
    transcript = os.path.join(hp.data, 'jss.v1.0.txt')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    if mode == "train":
        lines = lines[:-100]
    else:
        lines = lines[-100:]

    for line in lines:
        fname, text = line.strip().split("|")
        fpath = os.path.join(hp.data, fname)
        fpaths.append(fpath)

        text += "␃"  # ␃: EOS
        if hp.token_type == "char": # syllable
            text = list(text)
        else:
            text = [h2j(char) for char in text]
            text = chain.from_iterable(text)
            if hp.token_type == "j": # jamo
                text = [h2j(char) for char in text]
            elif hp.token_type == "sj":  # single jamo
                text = [j2sj.get(j, j) for j in text]
            elif hp.token_type == "hcj": # hangul compatibility jamo
                text = [j2hcj.get(j, j) for j in text]
            elif hp.token_type == "shcj": # single hangul compatibility jamo
                text = [j2shcj.get(j, j) for j in text]
        text = chain.from_iterable(text)

        text = [char2idx[char] for char in text if char in char2idx]
        text_lengths.append(len(text))
        if mode == "train":
            texts.append(np.array(text, np.int32).tostring())
        else:
            texts.append(text + [0]*(hp.max_N-len(text)))

    return fpaths, text_lengths, texts


def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)
        print("maxlen=", maxlen, "minlen=", minlen)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        fname, mel, mag, t = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32, tf.int64])
        gt, = tf.py_func(guided_attention, [text_length, t], [tf.float32])

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))
        gt.set_shape((hp.max_N, hp.max_T))

        # Batching
        _, (texts, mels, mags, gts, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, gt, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 40)],
                                            num_threads=8,
                                            capacity=hp.B*10,
                                            dynamic_pad=True)

    return texts, mels, mags, gts, fnames, num_batch

# -*- coding: utf-8 -*-
#/usr/bin/python

class Hyperparams:
    '''Hyper parameters'''
    token_type = "j" # char (character) | j (Hangul jamo) | sj (Hagul jamo single) | hcj (hangul compatibility jamo) | shcj (single hangul compatibility jamo)

    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model
    r = 4 # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = 512 # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = "../jss"

    if token_type == "char":
        vocab = "␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅨᆞᆢᆨᆫᆮᆯᆰᆱᆷᆸᆺᆼᆽᆾ" \
                "가각간갈감갑값갓갔강갖같개객갠갤갯갱거걱걲건걷걸검겁것겅겉겊게겐겔겜겟겠겡겨격겪견결겸겹겻경곁계" \
                "고곡곤곧골곰곱곳공곶과곽관괄광괘괜괭괴교굘구국군굳굴굵굶굼굽굿궁궂궈권궝궤궨궹귀귄귓규균귤그극근" \
                "귿글긁금급긋긔기긴길김깁깃깅까깍깎깐깔깜깝깟깡깥깨깻꺼꺽꺾껀껄껌껍껏껑께껜껠껫껭껴꼇꼉꼬꼭꼴꼼꼽" \
                "꼿꽁꽂꽃꽈꽉꽛꽝꽤꽹꾸꾹꾼꿀꿈꿔꿩꿰뀌뀐뀔끄끅끈끊끌끓끔끕끗끝끠끼낀낄낌낍나낙난날남납낫났낭낮낳" \
                "내낵낸낼냄냅냇냈냉냐냑냥너넉넌널넓넘넙넝넣네넥넨넷넹녀녁년녈념녑녓녕녘녜녠노녹논놀놈놉놋농높놓놔" \
                "놘놤놧놩뇌뇨누눅눈눌눔눕눙눠눵뉴늄느늑는늘늙늠능늦늬늴니닌닐님닙닝다닥닦단닫달닭닮담답닷당닿대댄" \
                "댈댓댕댜더덕던덜덟덤덥덧덩덮데덴델뎀뎃뎅뎌뎡도독돈돋돌돔돗동돠돵돼되된될됩두둑둔둘둠둡둥둬둰둿뒁" \
                "뒈뒌뒐뒘뒙뒛뒝뒤뒨뒷드득든듣들듬듭듯등듸듼딋디딘딜딧딪따딱딴딸땅때땐땔땡떠떡떤떨떰떳떵떻떼뗀뗌뗑" \
                "또똑똥뚜뚝뚤뚱뛰뛸뜨뜩뜬뜯뜰뜸뜻띄띠띤라락란람랍랏랐랑랗래랜랫랭랴략량러럭런럴럼럽럿렁렇레렉렌렙" \
                "렛렝려력련렬렴렵렷렸령렿례롄로록론롬롭롯롱롸뢈뢍료룐룔루룩룬룰룸룹룻룽뤌뤠류륜률륨르륵른를름릅릇" \
                "릉릎리린릴림립릿마막만많말맑맙맛망맞맡맣매맥맨맬맴맵맷맹머먹먼멀멈멋멍멓메멕멘멜멤멥멧멩멪멫며면" \
                "몃명몇몟모목몬몰몸못몽묏묘묫무묵묶문묻물뭄뭇뭉뭐뭔뭘뭣므믄믈믐믓미믹민믿밀밋밍밑바박밖반받발밟밤밥" \
                "밧방밭배백뱀뱃뱅버벅번벋벌범법벗벙벚베벡벤벨벳벵벼벽변별볏병볕보복볶본볼볿봄봅봇봉봐봔봠봣봤봥봬부" \
                "북분불붉붐붑붓붕붙붸브븐블비빈빌빕빗빙빛빠빡빤빨빵빼뺏뺑뻐뻔뻘뻬뻿뼝뽀뽄뽐뽑뽕뿌뿐뿔뿜쁘쁜삐삔삘삣" \
                "삥사삭산살삶삼삽삿상새색샌샐샛생샤서석섞선섣설섬섭섯성섶세섹센셀셉셋셍셔션셤셧셨셰소속손솔솜솟송솥" \
                "솨쇠쇼수숙순숟술숨숫숭숯숲숸쉐쉔쉘쉥쉬쉰쉽슈스슨슬슴습슷승싀싓시식신싣실싫심십싯싱싶싸싹싼쌀쌂쌈쌉" \
                "쌋쌍쌓쌔쌩써썩썬썰썸썹썻썽쎄쎅쎈쏘쏙쏠쏴쑤쑥쑨쑬쑵쑹쒀쒄쒓쒕쒜쒯쓰쓴쓸씀씁씌씨씩씬씰씸씹씻아악안앉" \
                "않알암압앗앙앚앞애액야약얀얄얇얌얍양얕얗얘어억언얹얻얼얽엄업없엇었엉엌엎에엔엘엣엥여역엮연열염엽엿" \
                "였영옆예옌옛옝오옥온올옮옳옴옵옷옹옻와왁완왈왐왓왔왕왜외요욕욜용우욱운울움웁웃웅워웍원월웜웟웠웡웨" \
                "웬웸웻웽위윌윗유육윤율윷으은을음읍응의읜읠이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재잿쟁저적전" \
                "절젊점접젓정젖제젠젤젭젯젱져젼졈졋졌졍조족존졸좀좁종좋좌죄주죽준줄줌줍줏중줘줙줜줨줫줬줭줴쥐쥑즈즉" \
                "즌즐즘증지직진짇질짊짐집짓징짚짛짜짝짠짤짧짬짭짱째쨋쨌쩌쩍쩐쩡쩨쩬쪄쪈쪗쪙쪼쪽쫄쫌쫑쫙쬐쭈쭉쭐쭤쮀" \
                "쯤찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻창찾채책챈처척천철첨첩첫청체첵첸쳇쳉쳐쳔쳣쳤쳥초촉촌촐촘촛총촤" \
                "촷촹최추축춘출춤춥충춰췅췌취측츰층치칙친칠칡침칩칫칭카칸칼캅캉캐커컨컬컴컵컷컸컹케켄켈켜켠켯켰켱코" \
                "콕콘콜콤콥콩쿠쿡쿤쿨쿵퀘퀴큇큐크큰클큼큽키킬킴킵킹타탁탄탈탐탑탓탕태택탯탱터턱턴털텀텁텃텅테텐텔텝" \
                "텟텡토톡톤톧톨톰톱톳통톼퇏퇴투툭툰퉁퉤퉷튀트특튼튿틀틈틉틑틔티파팍판팔팝팟팡팥패팬팽퍼퍽펌펏펑페펜" \
                "펭펴편평폐포폭폰폴폽퐁표푸푹푼풀품풉풍프픈플픕피픽핀필핏하학한할함합핫항해핸핼햇했행향허헌헐험헙헛" \
                "헝헤헨헴헷헸헹혀현혈협혓형혜호혹혼홀홈홉홋홍화확환활황회효횹후훅훈훌훍훑훗훤훼휄휏휘휠휴흐흑흔흘흙" \
                "흡희흰흴히힌힐힘힝"
        max_N, max_T = 106, 380
    elif token_type == "j":
        vocab = '␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲ' \
                'ᅳᅴᅵᆞᆢᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'
        max_N, max_T = 220, 380
    elif token_type == "sj":
        vocab = '␀␃ !,.?ᄀᄂᄃᄅᄆᄇᄉᄋᄌᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲ' \
                'ᅳᅴᅵᆞᆢᆨᆫᆮᆯᆷᆸᆺᆼᆽᆾᆿᇀᇁᇂ'
        max_N, max_T = 220, 380
    elif token_type == "hcj":
        vocab = "␀␃ !,.?ㄱㄲㄴㄵㄶㄷㄸㄹㄺㄻㄼㄾㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㆍᆢ"
        max_N, max_T = 220, 380
    elif token_type == "shcj":
        vocab = "␀␃ !,.?ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㆍᆢ"
        max_N, max_T = 220, 380


    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/{}".format(token_type)
    sampledir = 'samples/{}'.format(token_type)
    B = 8 # batch size
    num_iterations = 200000

# -*- coding: utf-8 -*-

from tqdm import tqdm
import soundfile as sf
import pysptk
import pyworld
from nnmnkwii.preprocessing.alignment import DTWAligner
import nnmnkwii.metrics

aligner = DTWAligner()

def get_mc(wav):
    y, sr = sf.read(wav)
    y = y.astype(np.float64)
    f0, timeaxis = pyworld.dio(y, sr, frame_period=5)
    f0 = pyworld.stonemask(y, f0, timeaxis, sr)
    spectrogram = pyworld.cheaptrick(y, f0, timeaxis, sr)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=0.41)
    mc = mc.astype(np.float32)

    return mc


def get_mcd(inp, ref):
    # extract mc
    inp_mc = get_mc(inp)
    ref_mc = get_mc(ref)

    # alignment
    inp = np.expand_dims(inp_mc, 0) # rank=3
    ref = np.expand_dims(ref_mc, 0) # rank=3

    inp_aligned, ref_aligned = aligner.transform((inp, ref))

    inp_aligned = np.squeeze(inp_aligned)
    ref_aligned = np.squeeze(ref_aligned)

    # calc mcd
    mcd = nnmnkwii.metrics.melcd(inp_aligned, ref_aligned)

    return mcd


if __name__ == "__main__":
    def run(token_type):
        mcd_li = []
        for i in tqdm(range(1, 101)):
            inp = 'samples/{}/{}.wav'.format(token_type, i)
            ref = '/data/public/rw/jss/jss/{}.wav'.format(9900-1+i)
            mcd = get_mcd(inp, ref)
            mcd_li.append(mcd)
        mcd_li = np.array(mcd_li)
        print('{}'.format(token_type))
        print('mean =', mcd_li.mean())
        print('var =', mcd_li.var())

    # run("char")
    # run("j")
    # run("hcj")
    # run("shcj")
    run("sj")

# -*- coding: utf-8 -*-


import tensorflow as tf


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs


def normalize(inputs,
              scope="normalize",
              reuse=None):
    '''Applies layer normalization that normalizes along the last axis.
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over the last dimension.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    outputs = tf.contrib.layers.layer_norm(inputs,
                                           begin_norm_axis=-1,
                                           scope=scope,
                                           reuse=reuse)
    return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387
    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        outputs = H * T + inputs * (1. - T)
    return outputs


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           dropout_rate=0,
           use_bias=True,
           activation_fn=None,
           training=True,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      dropout_rate: A float of [0, 1].
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        tensor = normalize(tensor)
        if activation_fn is not None:
            tensor = activation_fn(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor


def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0,
       use_bias=True,
       activation_fn=None,
       training=True,
       scope="hc",
       reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    _inputs = inputs
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": 2 * filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        H1, H2 = tf.split(tensor, 2, axis=-1)
        H1 = normalize(H1, scope="H1")
        H2 = normalize(H2, scope="H2")
        H1 = tf.nn.sigmoid(H1, "gate")
        H2 = activation_fn(H2, "info") if activation_fn is not None else H2
        tensor = H1 * H2 + (1. - H1) * _inputs

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor


def conv1d_transpose(inputs,
                     filters=None,
                     size=3,
                     stride=2,
                     padding='same',
                     dropout_rate=0,
                     use_bias=True,
                     activation=None,
                     training=True,
                     scope="conv1d_transpose",
                     reuse=None):
    '''
        Args:
          inputs: A 3-D tensor with shape of [batch, time, depth].
          filters: An int. Number of outputs (=activation maps)
          size: An int. Filter size.
          rate: An int. Dilation rate.
          padding: Either `same` or `valid` or `causal` (case-insensitive).
          dropout_rate: A float of [0, 1].
          use_bias: A boolean.
          activation_fn: A string.
          training: A boolean. If True, dropout is applied.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A tensor of the shape with [batch, time*2, depth].
        '''
    with tf.variable_scope(scope, reuse=reuse):
        if filters is None:
            filters = inputs.get_shape().as_list()[-1]
        inputs = tf.expand_dims(inputs, 1)
        tensor = tf.layers.conv2d_transpose(inputs,
                                            filters=filters,
                                            kernel_size=(1, size),
                                            strides=(1, stride),
                                            padding=padding,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            use_bias=use_bias)
        tensor = tf.squeeze(tensor, 1)
        tensor = normalize(tensor)
        if activation is not None:
            tensor = activation(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor

# -*- coding: utf-8 -*-


from hyperparams import Hyperparams as hp
from modules import *


def TextEnc(L, training=True):
    '''
    Args:
      L: Text inputs. (B, N)
    Return:
        K: Keys. (B, N, d)
        V: Values. (B, N, d)
    '''
    i = 1
    tensor = embed(L,
                   vocab_size=len(hp.vocab),
                   num_units=hp.e,
                   scope="embed_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    filters=2*hp.d,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            activation_fn=None,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    K, V = tf.split(tensor, 2, -1)
    return K, V

def AudioEnc(S, training=True):
    '''
    Args:
      S: melspectrogram. (B, T/r, n_mels)
    Returns
      Q: Queries. (B, T/r, d)
    '''
    i = 1
    tensor = conv1d(S,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            padding="CAUSAL",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=3,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    return tensor

def Attention(Q, K, V, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)
      mononotic_attention: A boolean. At training, it is False.
      prev_max_attentions: (B,). At training, it is set to None.
    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      alignments: (B, N, T/r)
      max_attentions: (B, T/r)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.rsqrt(tf.to_float(hp.d))
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions-1, hp.max_N)
        reverse_masks = tf.sequence_mask(hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)
    A = tf.nn.softmax(A) # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)

    return R, alignments, max_attentions

def AudioDec(R, training=True):
    '''
    Args:
      R: [Context Vectors; Q]. (B, T/r, 2d)
    Returns:
      Y: Melspectrogram predictions. (B, T/r, n_mels)
    '''

    i = 1
    tensor = conv1d(R,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(4):
        tensor = hc(tensor,
                        size=3,
                        rate=3**j,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    for _ in range(3):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        padding="CAUSAL",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    # mel_hats
    logits = conv1d(tensor,
                    filters=hp.n_mels,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    Y = tf.nn.sigmoid(logits) # mel_hats

    return logits, Y

def SSRN(Y, training=True):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)
    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''

    i = 1 # number of layers

    # -> (B, T/r, c)
    tensor = conv1d(Y,
                    filters=hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(2):
        tensor = hc(tensor,
                      size=3,
                      rate=3**j,
                      dropout_rate=hp.dropout_rate,
                      training=training,
                      scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor,
                                  scope="D_{}".format(i),
                                  dropout_rate=hp.dropout_rate,
                                  training=training,); i += 1
        for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    # -> (B, T, 2*c)
    tensor = conv1d(tensor,
                    filters=2*hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    # -> (B, T, 1+n_fft/2)
    tensor = conv1d(tensor,
                    filters=1+hp.n_fft//2,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    logits = conv1d(tensor,
               size=1,
               rate=1,
               dropout_rate=hp.dropout_rate,
               training=training,
               scope="C_{}".format(i))
    Z = tf.nn.sigmoid(logits)
    return logits, Z

# -*- coding: utf-8 -*-

import os

from hyperparams import Hyperparams as hp
import numpy as np
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm


def synthesize():
    # Load data
    _, _, L = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)

if __name__ == '__main__':
    synthesize()
    print("Done")

# -*- coding: utf-8 -*-


from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys


class Graph:
    def __init__(self, num=1, mode="train"):
        '''
        Args:
          num: Either 1 or 2. 1 for Text2Mel 2 for SSRN.
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## L: Text. (B, N), int32
        ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        ## mags: Magnitude. (B, T, n_fft//2+1) float32
        if mode=="train":
            self.L, self.mels, self.mags, self.gts, self.fnames, self.num_batch = get_batch()
            self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            # self.gts = tf.convert_to_tensor(guided_attention())
        else:  # Synthesize
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        if num==1 or (not training):
            with tf.variable_scope("Text2Mel"):
                # Get S or decoder inputs. (B, T//r, n_mels)
                self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

                # Networks
                with tf.variable_scope("TextEnc"):
                    self.K, self.V = TextEnc(self.L, training=training)  # (N, Tx, e)

                with tf.variable_scope("AudioEnc"):
                    self.Q = AudioEnc(self.S, training=training)

                with tf.variable_scope("Attention"):
                    # R: (B, T/r, 2d)
                    # alignments: (B, N, T/r)
                    # max_attentions: (B,)
                    self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                             mononotic_attention=(not training),
                                                                             prev_max_attentions=self.prev_max_attentions)
                with tf.variable_scope("AudioDec"):
                    self.Y_logits, self.Y = AudioDec(self.R, training=training) # (B, T/r, n_mels)
        else:  # num==2 & training. Note that during training,
            # the ground truth melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(self.mels, training=training)

        if not training:
            # During inference, the predicted melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(self.Y, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if training:
            if num==1: # Text2Mel
                # mel L1 loss
                self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

                # mel binary divergence loss
                self.loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.mels))

                # guided_attention loss
                self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
                # self.attention_masks = tf.to_float(tf.not_equal(self.gts, -1))
                self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
                self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)
                self.mask_sum = tf.reduce_sum(self.attention_masks)
                self.loss_att /= self.mask_sum

                # total loss
                self.loss = self.loss_mels + self.loss_bd1 + self.loss_att

                tf.summary.scalar('train/loss_mels', self.loss_mels)
                tf.summary.scalar('train/loss_bd1', self.loss_bd1)
                tf.summary.scalar('train/loss_att', self.loss_att)
                tf.summary.image('train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
                tf.summary.image('train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
            else: # SSRN
                # mag L1 loss
                self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

                # mag binary divergence loss
                self.loss_bd2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.mags))

                # total loss
                self.loss = self.loss_mags + self.loss_bd2

                tf.summary.scalar('train/loss_mags', self.loss_mags)
                tf.summary.scalar('train/loss_bd2', self.loss_bd2)
                tf.summary.image('train/mag_gt', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
                tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))

            # Training Scheme
            self.lr = learning_rate_decay(hp.lr, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    num = int(sys.argv[1])

    g = Graph(num=num); print("Training Graph loaded")

    logdir = hp.logdir + "-" + str(num)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                    if num==1:
                        # plot alignment
                        alignments = sess.run(g.alignments)
                        gts = sess.run(g.gts)
                        plot_alignment(alignments[0], str(gs // 1000).zfill(3) + "k", logdir)
                        # plot_alignment(gts[0], gs, logdir)

            # break
            if gs > hp.num_iterations: break

    print("Done")

# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal

from hyperparams import Hyperparams as hp
import tensorflow as tf

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # # Trimming
    # y, _ = librosa.effects.trim(y, top_db=40)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav, top_db=40)
    # wav = trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.
    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')

def guided_attention(n, t, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    gt = np.ones((hp.max_N, hp.max_T), np.float32)
    for n_pos in range(n):
        for t_pos in range(t):
            gt[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(t) - n_pos / float(n)) ** 2 / (2 * g * g))

    return gt

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::hp.r, :]
    t = mel.shape[0]

    return fname, mel, mag, t

#This is adapted by
# https://github.com/keithito/tacotron/blob/master/util/audio.py#L55-62
def trim(wav, top_db=40, min_silence_sec=0.8):
    frame_length = int(hp.sr * min_silence_sec)
    hop_length = int(frame_length / 4)
    endpoint = librosa.effects.split(wav, frame_length=frame_length,
                               hop_length=hop_length,
                               top_db=top_db)[0, 1]
    return wav[:endpoint]

def load_j2hcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)
    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character
    '''
    j   = 'ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲ' \
      'ᅳᅴᅵᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂᆞ'
    hcj = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅇㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠ' \
          'ㅡㅢㅣㄱㄲㄴㄵㄶㄷㄹㄺㄻㄼㄾㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎㆍ'
    assert len(j) == len(hcj)
    j2hcj = {j_: hcj_ for j_, hcj_ in zip(j, hcj)}
    return j2hcj

def load_j2sj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)
    Returns:
      A dictionary that decomposes double consonants into two single consonants.
    '''
    j = 'ᄁᄄᄈᄊᄍᆩᆬᆭᆰᆱᆲᆴᆶᆹᆻ'
    sj = 'ᄀᄀ|ᄃᄃ|ᄇᄇ|ᄉᄉ|ᄌᄌ|ᆨᆨ|ᆫᆽ|ᆫᇂ|ᆯᆨ|ᆯᆷ|ᆯᆸ|ᆯᇀ|ᆯᇂ|ᆸᆺ|ᆺᆺ'
    assert len(j)==len(sj.split("|"))
    j2sj = {j_: sj_ for j_, sj_ in zip(j, sj.split("|"))}
    return j2sj

def load_j2shcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)
    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character.
      Double consonants are further decomposed into single consonants.
    '''
    j   = 'ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲ' \
      'ᅳᅴᅵᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂᆞ'
    shcj = 'ㄱ|ㄱㄱ|ㄴ|ㄷ|ㄷㄷ|ㄹ|ㅁ|ㅂ|ㅂㅂ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅈㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ|ㅇ|ㅏ|ㅐ|ㅑ|ㅒ|ㅓ|ㅔ|ㅕ|ㅖ|ㅗ|ㅘ|ㅙ|ㅚ|ㅛ|ㅜ|ㅝ|ㅞ|ㅟ|ㅠ|' \
    'ㅡ|ㅢ|ㅣ|ㄱ|ㄱㄱ|ㄴ|ㄴㅈ|ㄴㅎ|ㄷ|ㄹ|ㄹㄱ|ㄹㅁ|ㄹㅂ|ㄹㅌ|ㄹㅎ|ㅁ|ㅂ|ㅂㅅ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ|ㆍ'

    assert len(j)==len(shcj.split("|"))
    j2shcj = {j_: shcj_ for j_, shcj_ in zip(j, shcj.split("|"))}
    return j2shcj
