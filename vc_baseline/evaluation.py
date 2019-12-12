from nnmnkwii.metrics import melcd
import matplotlib.pyplot as plt
import librosa

X = []

for i in range(1, 55):
    fileA,_ = librosa.core.load('data/evaluation_all/SF1/2000' + '{:02}'.format(int(i)) + '.wav')
    fileB,_ = librosa.core.load('converted_voices/2000' + '{:02}'.format(int(i)) + '.wav')
    mfccA = librosa.feature.mfcc(fileA)
    mfccB = librosa.feature.mfcc(fileB)
    minLength = min(mfccA.shape[1], mfccB.shape[1])
    mfccA = mfccA[:, :minLength]
    mfccB = mfccB[:, :minLength]

    X.append(melcd(mfccA, mfccB))

plt.plot(X, 'ro')
plt.xlabel('File ID')
plt.ylabel('MCD')
plt.savefig('mcd.png')
print(sum(X)/len(X))
