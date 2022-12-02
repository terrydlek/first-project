from ModulationPy.ModulationPy import *
from ModulationPy import QAMModem
import numpy as np
import matplotlib.pyplot as plt

class Modem:
    def __init__(self, M, gray_map=True, bin_input=True, soft_decision=True, bin_output=True):

        N = np.log2(M)  # bits per symbol
        if N != np.round(N):
            raise ValueError("M should be 2**n, with n=1, 2, 3...")
        if soft_decision == True and bin_output == False:
            raise ValueError("Non-binary output is available only for hard decision")

        self.M = M  # modulation order
        self.N = int(N)  # bits per symbol
        self.m = [i for i in range(self.M)]
        self.gray_map = gray_map
        self.bin_input = bin_input
        self.soft_decision = soft_decision
        self.bin_output = bin_output

    def __gray_encoding(self, dec_in):
        bin_seq = [np.binary_repr(d, width=self.N) for d in dec_in]
        gray_out = []
        for bin_i in bin_seq:
            gray_vals = [str(int(bin_i[idx]) ^ int(bin_i[idx - 1]))
                         if idx != 0 else bin_i[0]
                         for idx in range(0, len(bin_i))]
            gray_i = "".join(gray_vals)
            gray_out.append(int(gray_i, 2))
        return gray_out

    def create_constellation(self, m, s):
        if self.bin_input == False and self.gray_map == False:
            dict_out = {k: v for k, v in zip(m, s)}
        elif self.bin_input == False and self.gray_map == True:
            mg = self.__gray_encoding(m)
            dict_out = {k: v for k, v in zip(mg, s)}
        elif self.bin_input == True and self.gray_map == False:
            mb = self.de2bin(m)
            dict_out = {k: v for k, v in zip(mb, s)}
        elif self.bin_input == True and self.gray_map == True:
            mg = self.__gray_encoding(m)
            mgb = self.de2bin(mg)
            dict_out = {k: v for k, v in zip(mgb, s)}
        return dict_out

    def llr_preparation(self):
        code_book = self.code_book

        zeros = [[] for i in range(self.N)]
        ones = [[] for i in range(self.N)]

        bin_seq = self.de2bin(self.m)

        for bin_idx, bin_symb in enumerate(bin_seq):
            if self.bin_input == True:
                key = bin_symb
            else:
                key = bin_idx
            for possition, digit in enumerate(bin_symb):
                if digit == '0':
                    zeros[possition].append(code_book[key])
                else:
                    ones[possition].append(code_book[key])
        return zeros, ones

    def __ApproxLLR(self, x, noise_var):
        zeros = self.zeros
        ones = self.ones
        LLR = []
        for (zero_i, one_i) in zip(zeros, ones):
            num = [((np.real(x) - np.real(z)) ** 2)
                   + ((np.imag(x) - np.imag(z)) ** 2)
                   for z in zero_i]
            denum = [((np.real(x) - np.real(o)) ** 2)
                     + ((np.imag(x) - np.imag(o)) ** 2)
                     for o in one_i]

            num_post = np.amin(num, axis=0, keepdims=True)
            denum_post = np.amin(denum, axis=0, keepdims=True)

            llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
            LLR.append(-llr / noise_var)

        result = np.zeros((len(x) * len(zeros)))
        for i, llr in enumerate(LLR):
            result[i::len(zeros)] = llr
        return result

    def modulate(self, msg):
        if (self.bin_input == True) and ((len(msg) % self.N) != 0):
            raise ValueError("The length of the binary input should be a multiple of log2(M)")

        if (self.bin_input == True) and ((max(msg) > 1.) or (min(msg) < 0.)):
            raise ValueError("The input values should be 0s or 1s only!")
        if (self.bin_input == False) and ((max(msg) > (self.M - 1)) or (min(msg) < 0.)):
            raise ValueError("The input values should be in following range: [0, ... M-1]!")

        if self.bin_input:
            msg = [str(bit) for bit in msg]
            splited = ["".join(msg[i:i + self.N])
                       for i in range(0, len(msg), self.N)]  # subsequences of bits
            modulated = [self.code_book[s] for s in splited]
        else:
            modulated = [self.code_book[dec] for dec in msg]
        return np.array(modulated)

    def demodulate(self, x, noise_var=1.):
        if self.soft_decision:
            result = self.__ApproxLLR(x, noise_var)
        else:
            if self.bin_output:
                llr = self.__ApproxLLR(x, noise_var)
                result = (np.sign(-llr) + 1) / 2  # NRZ-to-bin
            else:
                llr = self.__ApproxLLR(x, noise_var)
                result = self.bin2de((np.sign(-llr) + 1) / 2)
        return result

class QAMModem(Modem):
    def __init__(self, M, gray_map=True, bin_input=True, soft_decision=True, bin_output=True):
        super().__init__(M, gray_map, bin_input, soft_decision, bin_output)

        if np.sqrt(M) != np.fix(np.sqrt(M)) or np.log2(np.sqrt(M)) != np.fix(np.log2(np.sqrt(M))):
            raise ValueError('M must be a square of a power of 2')

        self.m = [i for i in range(self.M)]
        self.s = self.__qam_symbols()
        self.code_book = self.create_constellation(self.m, self.s)

        if self.gray_map:
            self.__gray_qam_arange()

        self.zeros, self.ones = self.llr_preparation()

    def __qam_symbols(self):
        c = np.sqrt(self.M)
        b = -2 * (np.array(self.m) % c) + c - 1
        a = 2 * np.floor(np.array(self.m) / c) - c + 1
        s = list((a + 1j * b))
        return s

    def __gray_qam_arange(self):
        for idx, (key, item) in enumerate(self.code_book.items()):
            if (np.floor(idx / np.sqrt(self.M)) % 2) != 0:
                self.code_book[key] = np.conj(item)

    def de2bin(self, decs):
        bin_out = [np.binary_repr(d, width=self.N) for d in decs]
        return bin_out

    def bin2de(self, bin_in):
        dec_out = []
        N = self.N  # bits per modulation symbol (local variables are tiny bit faster)
        Ndecs = int(len(bin_in) / N)  # length of the decimal output
        for i in range(Ndecs):
            bin_seq = bin_in[i * N:i * N + N]  # binary equivalent of the one decimal value
            str_o = "".join([str(int(b)) for b in bin_seq])  # binary sequence to string
            dec_out.append(int(str_o, 2))
        return dec_out

    def plot_const(self):
        if self.M <= 16:
            limits = np.log2(self.M)
            size = 'small'
        elif self.M == 64:
            limits = 1.5 * np.log2(self.M)
            size = 'x-small'
        else:
            limits = 2.25 * np.log2(self.M)
            size = 'xx-small'

        const = self.code_book
        fig = plt.figure(figsize=(6, 4), dpi=150)
        for i in list(const):
            x = np.real(const[i])
            y = np.imag(const[i])
            plt.plot(x, y, 'o', color='red')
            if x < 0:
                h = 'right'
                xadd = -.05
            else:
                h = 'left'
                xadd = .05
            if y < 0:
                v = 'top'
                yadd = -.05
            else:
                v = 'bottom'
                yadd = .05
            if abs(x) < 1e-9 and abs(y) > 1e-9:
                h = 'center'
            elif abs(x) > 1e-9 and abs(y) < 1e-9:
                v = 'center'
            plt.annotate(i, (x + xadd, y + yadd), ha=h, va=v, size=size)
        M = str(self.M)
        if self.gray_map:
            mapping = 'Gray'
        else:
            mapping = 'Binary'

        if self.bin_input:
            inputs = 'Binary'
        else:
            inputs = 'Decimal'

        plt.grid()
        plt.axvline(linewidth=1.0, color='black')
        plt.axhline(linewidth=1.0, color='black')
        plt.axis([-limits, limits, -limits, limits])
        plt.title(M + '-QAM, Mapping: ' + mapping + ', Input: ' + inputs)
        plt.show()

modem = QAMModem(64,
                 gray_map=True,
                 bin_input=False)

modem.plot_const()

modem = PSKModem(16,
                 bin_input=False,
                 soft_decision=False,
                 bin_output=False)

msg = np.array([i for i in range(16)]) # input message

modulated = modem.modulate(msg) # modulation
demodulated = modem.demodulate(modulated) # demodulation
print("Demodulated message:\n"+str(modulated))
print("Demodulated message:\n"+str(demodulated))

from ModulationPy import QAMModem
from scipy import special


def BER_calc(a, b):
    num_ber = np.sum(np.abs(a - b))
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber


def BER_qam(M, EbNo):
    EbNo_lin = 10 ** (EbNo / 10)
    if M > 4:
        P = 2 * np.sqrt((np.sqrt(M) - 1) /
                        (np.sqrt(M) * np.log2(M))) * special.erfc(np.sqrt(EbNo_lin * 3 * np.log2(M) / 2 * (M - 1)))
    else:
        P = 0.5 * special.erfc(np.sqrt(EbNo_lin))
    return P


EbNos = np.array([i for i in range(30)])  # array of Eb/No in dBs
N = 100000  # number of symbols per the frame
N_c = 100  # number of trials

Ms = [4, 16, 64, 256]  # modulation orders

mean_BER = np.empty((len(EbNos), len(Ms)))
for idxM, M in enumerate(Ms):
    print("Modulation order: ", M)
    BER = np.empty((N_c,))
    k = np.log2(M)  # number of bit per modulation symbol

    modem = QAMModem(M,
                     bin_input=True,
                     soft_decision=False,
                     bin_output=True)

    for idxEbNo, EbNo in enumerate(EbNos):
        print("EbNo: ", EbNo)
        snrdB = EbNo + 10 * np.log10(k)  # Signal-to-Noise ratio (in dB)
        noiseVar = 10 ** (-snrdB / 10)  # noise variance (power)

        for cntr in range(N_c):
            message_bits = np.random.randint(0, 2, int(N * k))  # message
            modulated = modem.modulate(message_bits)  # modulation

            Es = np.mean(np.abs(modulated) ** 2)  # symbol energy
            No = Es / ((10 ** (EbNo / 10)) * np.log2(M))  # noise spectrum density

            noisy = modulated + np.sqrt(No / 2) * \
                    (np.random.randn(modulated.shape[0]) +
                     1j * np.random.randn(modulated.shape[0]))  # AWGN

            demodulated = modem.demodulate(noisy, noise_var=noiseVar)
            NumErr, BER[cntr] = BER_calc(message_bits,
                                         demodulated)  # bit-error ratio
        mean_BER[idxEbNo, idxM] = np.mean(BER, axis=0)  # averaged bit-error ratio

BER_theor = np.empty((len(EbNos), len(Ms)))
for idxM, M in enumerate(Ms):
    BER_theor[:, idxM] = BER_qam(M, EbNos)

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

plt.semilogy(EbNos, BER_theor[:, 0], 'g-', label='4-QAM (theory)')
plt.semilogy(EbNos, BER_theor[:, 1], 'b-', label='16-QAM (theory)')
plt.semilogy(EbNos, BER_theor[:, 2], 'k-', label='64-QAM (theory)')
plt.semilogy(EbNos, BER_theor[:, 3], 'r-', label='256-QAM (theory)')

plt.semilogy(EbNos, mean_BER[:, 0], 'g-o', label='4-QAM (simulation)')
plt.semilogy(EbNos, mean_BER[:, 1], 'b-o', label='16-QAM (simulation)')
plt.semilogy(EbNos, mean_BER[:, 2], 'k-o', label='64-QAM (simulation)')
plt.semilogy(EbNos, mean_BER[:, 3], 'r-o', label='256-QAM (simulation)')

ax.set_ylim(1e-7, 2)
ax.set_xlim(0, 25.1)

plt.title("M-QAM")
plt.xlabel('EbNo (dB)')
plt.ylabel('BER')
plt.grid()
plt.legend(loc='upper right')
plt.savefig('qam_ber.png')
