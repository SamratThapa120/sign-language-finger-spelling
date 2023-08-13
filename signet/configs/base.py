import numpy as np
class Base:
    char_to_idx = {" ":0,"!":1,"#":2,"$":3,"%":4,"&":5,"'":6,"(":7,")":8,"*":9,"+":10,",":11,"-":12,".":13,"/":14,"0":15,"1":16,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":24,":":25,";":26,"=":27,"?":28,"@":29,"[":30,"_":31,"a":32,"b":33,"c":34,"d":35,"e":36,"f":37,"g":38,"h":39,"i":40,"j":41,"k":42,"l":43,"m":44,"n":45,"o":46,"p":47,"q":48,"r":49,"s":50,"t":51,"u":52,"v":53,"w":54,"x":55,"y":56,"z":57,"~":58}
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    ROWS_PER_FRAME = 543
    MAX_WORD_LENGTH  = 64
    NUM_CLASSES = 60
    PAD = (-186.1324, np.int64(-1))  # padding value. First is padding value for input sequence, and second for prediction sequence.
    NOSE=[
        1,2,98,327
    ]
    LNOSE = [98]
    RNOSE = [327]
    LIP = [ 0, 
        61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
    RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

    POSE = [500, 502, 504, 501, 503, 505, 512, 513]
    LPOSE = [513,505,503,501]
    RPOSE = [512,504,502,500]

    REYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    ]
    LEYE = [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    ]

    LHAND = np.arange(468, 489).tolist()
    RHAND = np.arange(522, 543).tolist()
    CONNECTIONS=[
        [468, 469, 470, 471, 472],
        [468, 473, 474, 475, 476],
        [468, 477, 478, 479, 480],
        [468, 481, 482, 483, 484],
        [468, 485, 486, 487, 488],
        [497, 495, 494, 493, 489, 490, 491, 492, 496],
        [499, 498],
        [511, 505, 509, 507, 505, 503, 501, 500, 502, 504, 506, 508, 504, 510],
        [501, 513, 515, 517, 519, 521, 517],
        [500, 512, 514, 516, 518, 520, 516],
        [513, 512],
        [522, 523, 524, 525, 526],
        [522, 527, 528, 529, 530],
        [522, 531, 532, 533, 534],
        [522, 535, 536, 537, 538],
        [522, 539, 540, 541, 542]
    ]
    angle_a,angle_b,angle_c = [],[],[]
    for connection in CONNECTIONS:
        for i in range(len(connection) - 2):
            angle_a.append(connection[i])
            angle_b.append(connection[i+1])
            angle_c.append(connection[i+2])
    length_a,length_b = [],[]
    for connection in CONNECTIONS:
        for i in range(len(connection) - 1):
            length_a.append(connection[i])
            length_b.append(connection[i+1])
    useangle=False
    uselengths=False
    POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE #+POSE

    NUM_NODES = len(POINT_LANDMARKS)
    CHANNELS = 6*NUM_NODES
    combine_tensors=False
    combine_tensors_probability=0
    loss_type="ctc"
    attention_span=0
    lookahead=True
    use_depth=True
    NUM_ANGLES=sum([len(x)-2 for x in CONNECTIONS])
    NUM_LENGTHS=sum([len(x)-1 for x in CONNECTIONS])

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))}