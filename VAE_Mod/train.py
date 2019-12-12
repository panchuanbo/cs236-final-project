import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from preprocess import *
from model import CVAE

def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, val_A_dir, val_B_dir, output_dir, tensorboard_dir, load_path, gen_eval=True):
    np.random.seed(random_seed)

    # For now, copy hyperparams used in the CycleGAN
    num_epochs = 100000
    mini_batch_size = 1 # mini_batch_size = 1 is better
    learning_rate = 0.0002
    learning_rate_decay = learning_rate / 200000
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128
    lambda_cycle = 10
    lambda_identity = 5
    device = 'cuda'

    # Use the same pre-processing as the CycleGAN
    print("Begin Preprocessing")

    wavs_A = load_wavs(wav_dir=train_A_dir, sr = sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr = sampling_rate)
    print("Finished Loading")

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    print("Finished Encoding")

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))

    coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    print("Input data fixed.")
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)

    if val_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if val_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    print("End Preprocessing")

    if load_path is not None:
        model = CVAE(num_mcep, 128, num_mcep, 2)
        model.load_state_dict(torch.load(load_path))
        model.eval()
        if device == 'cuda':
            model.cuda()
        print("Loaded Model from path %s" % load_path)
        if val_A_dir is not None and gen_eval:
            print("Generating Evaluation Data")
            for file in os.listdir(val_A_dir):
                filepath = os.path.join(val_A_dir, file)
                print("Converting {0} from Class 0 to Class 1".format(filepath))
                wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                coded_sp_converted_norm, _, _ = model.convert(np.array([coded_sp_norm]), 0, 1, device)
                coded_sp_converted_norm = coded_sp_converted_norm.cpu().numpy()
                coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
                coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                librosa.output.write_wav(os.path.join(validation_A_output_dir, 'eval_' + os.path.basename(file)), wav_transformed, sampling_rate)
            exit(0)

    print("Begin Training")

    model = CVAE(num_mcep, 128, num_mcep, 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(tensorboard_dir)

    if device == 'cuda':
        model.cuda()

    for epoch in tqdm(range(num_epochs)):
        dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)
        dataset_A = torch.tensor(dataset_A).to(torch.float)
        dataset_B = torch.tensor(dataset_B).to(torch.float)

        n_samples, input_dim, depth = dataset_A.shape
        y_A = F.one_hot(torch.zeros(depth).to(torch.int64), num_classes=2).to(torch.float).T
        y_B = F.one_hot(torch.ones(depth).to(torch.int64), num_classes=2).to(torch.float).T
        (y_A, y_B) = (y_A.reshape((1, 2, depth)), y_B.reshape((1, 2, depth)))
        y_A = torch.cat([y_A]*n_samples)
        y_B = torch.cat([y_B]*n_samples)

        # dataset_A = torch.cat((dataset_A, y_A), axis=1)
        # dataset_B = torch.cat((dataset_B, y_B), axis=1)

        X = torch.cat((dataset_A, dataset_B)).to(device)
        Y = torch.cat((y_A, y_B)).to(device)

        # out, z_mu, z_var = model(dataset_A, y_A)
        # rec_loss = F.binary_cross_entropy(out, dataset_A, size_average=False)
        # kl_diver = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp())
        out, z_mu, z_var = model(X, Y)

        rec_loss = F.binary_cross_entropy(out, X, size_average=False)
        kl_diver = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp())

        loss = rec_loss + kl_diver

        writer.add_scalar('Reconstruction Loss', rec_loss, epoch)
        writer.add_scalar('KL-Divergence', kl_diver, epoch)
        writer.add_scalar('Total Loss', loss, epoch)

        # print("loss = {0} || rec = {1} || kl = {2}".format(loss, rec_loss, kl_diver))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if val_A_dir is not None:
            if epoch % 1000 == 0:
                print('Generating Validation Data...')
                for file in os.listdir(val_A_dir):
                    filepath = os.path.join(val_A_dir, file)
                    print("Converting {0} from Class 0 to Class 1".format(filepath))
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    coded_sp_converted_norm, _, _ = model.convert(np.array([coded_sp_norm]), 0, 1, device)
                    coded_sp_converted_norm = coded_sp_converted_norm.cpu().numpy()
                    coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, str(epoch) + '_' + os.path.basename(file)), wav_transformed, sampling_rate)
                    break
        if epoch % 1000 == 0:
            print('Saving Checkpoint')
            filepath = os.path.join(model_dir, model_name)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            torch.save(model.state_dict(), os.path.join(filepath, '{0}.ckpt'.format(epoch)))

train(
    train_A_dir='data/vcc2016_training/SF1',
    train_B_dir='data/vcc2016_training/TM1',
    model_dir='models',
    model_name='modified_cvae',
    random_seed=42,
    val_A_dir='data/vcc2016_evaluation/SF1',
    val_B_dir=None,
    output_dir='output',
    tensorboard_dir='tensorboard',
    load_path=None# 'models/basic_cvae/18000.ckpt'
)
