import io
import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot
from PIL import Image
from scipy.interpolate import splev, splprep
from torch.distributions import bernoulli

matplotlib.use("AGG")

logger = logging.getLogger()


class Scribe(object):
    """Generates an image of a word using the handwriting synthesis model and applies any requested augmentations at the word level"""

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_path = Path(__file__).parent / "data" / "scribe" / "original"
        self.model_path = self.data_path / "best_model_synthesis.pt"
        self.batch_size = 1
        self.bias = 10.0
        self.inp = torch.zeros(self.batch_size, 1, 3).to(self.device)

        self._load_model(self.model_path)

    def _load_model(self, weights: Path) -> None:
        """Loads model from given state_dict file"""
        with open(Path(__file__).parent / "data" / "scribe" / "original" / "char_to_id.pkl", "rb") as pkl:
            self.char_to_id = pickle.load(pkl)

        self.model = HandWritingSynthesisNet(window_size=len(self.char_to_id))

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.hidden, self.window_vector, self.kappa = self.model.init_hidden(self.batch_size, self.device)

    def generate_sequence(self, text: str) -> Tuple[Image.Image, str]:
        """Generates a word image from the given text. Requires text string to be at least 3 characters long."""

        glyphs = np.array(list(text + "  "))

        logger.info("".join(glyphs))

        glyphs = np.array([[self.char_to_id[char] for char in glyphs] for i in range(self.batch_size)]).astype(
            np.float32
        )
        glyphs = torch.from_numpy(glyphs).to(self.device)
        text_mask = torch.ones(glyphs.shape).to(self.device)

        logger.info("Generating sequence....")
        seq_len = 0
        gen_seq = []

        _hidden = self.hidden
        _kappa = self.kappa
        _inp = self.inp
        _window_vector = self.window_vector

        with torch.no_grad():
            batch_size = _inp.shape[0]
            logger.info(f"batch_size: {batch_size}")

            while not self.model.EOS and seq_len < 2000:
                y_hat, state, _window_vector, _kappa = self.model.forward(
                    _inp, glyphs, text_mask, _hidden, _window_vector, _kappa
                )

                __hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                _hidden = (__hidden, _cell)
                y_hat = y_hat.squeeze()
                Z = sample_from_out_dist(y_hat, self.bias)
                _inp = Z
                gen_seq.append(Z)

                seq_len += 1

        generated_sequence = torch.cat(gen_seq, dim=1)
        generated_sequence = generated_sequence.cpu().numpy()

        # NOTE 0.0 = train_mean and 1.0 = train_std
        generated_sequence = data_denormalization(0.0, 1.0, generated_sequence)

        return plot_stroke(generated_sequence[0]), self.model_path.stem


def plot_stroke(stroke: np.ndarray, print_size: int = 6) -> Image.Image:
    """Converts a stroke of a pen (path trace polypoints) into a image with transparent background"""
    f, ax = pyplot.subplots()

    # Calculate the cumulative sum of the elements in x and y columns
    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    # Find the range of x and y points
    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0

    f.set_size_inches(5.0 * size_x / size_y, 5.0)

    # Store the pen lift off points (1 = True)
    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        # Get the x and y values between pen lift off points
        arr_x = x[start:cut_value]
        arr_y = y[start:cut_value]

        # Create an empty array to store the above x and y values in the right format for interpolation
        arr_x_fl = np.empty([cut_value - start], dtype=float)
        arr_y_fl = np.empty([cut_value - start], dtype=float)

        # Re-assign the above x and y values to arr_x_fl and arr_y_fl for interpolation
        for i in range(len(arr_x)):
            arr_x_fl[i] = arr_x[i]
        for i in range(len(arr_y_fl)):
            arr_y_fl[i] = arr_y[i]

        # Interpolation between the x and y points for nicer plots
        try:
            if arr_x_fl.shape[0] != 0:
                tck, u = splprep([arr_x_fl, arr_y_fl], s=8, per=False, k=2)
                xi, yi = splev(np.linspace(0, 1, 60), tck)
                ax.plot(xi, yi, "k-", linewidth=print_size, antialiased=True)
        except Exception as e:
            logger.exception(f"Could not interpolate because: {e}")
            ax.plot(x[start:cut_value], y[start:cut_value], "k-", linewidth=print_size, antialiased=True)

        start = cut_value + 1

    ax.axis("off")  # equal
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    buf = io.BytesIO()
    f.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    pyplot.close()
    return img


def data_denormalization(mean: float, std: float, data: np.ndarray) -> np.ndarray:
    """
       Data denormalization using train set mean and std
    """
    data[:, :, 1:] *= std
    data[:, :, 1:] += mean
    return data


def stable_softmax(X: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """ Normalizes the output of the network to a probability distribution (without nan or None). """
    max_vec = torch.max(X, dim, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, dim, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat


def sample_from_out_dist(y_hat: torch.Tensor, bias: float) -> torch.Tensor:
    """ From the Neural Network output, selects a single option within the range set by the bias """
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = torch.sigmoid(y[0])
    # print(y[1] * (1 + bias))
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = (
        correlations[K] * std_1[K] * std_2[K],
        correlations[K] * std_1[K] * std_2[K],
    )

    x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(y_hat.device)
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample


class HandWritingSynthesisNet(nn.Module):
    """ Neural Network that generates quazi-handwriting word images. """

    def __init__(self, hidden_size: int = 400, n_layers: int = 3, output_size: int = 121, window_size: int = 77):
        super(HandWritingSynthesisNet, self).__init__()
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        K: int = 10
        self.EOS = False

        self.lstm_1 = nn.LSTM(3 + self.vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + self.vocab_size + hidden_size, hidden_size, batch_first=True)

        self.lstm_3 = nn.LSTM(3 + self.vocab_size + hidden_size, hidden_size, batch_first=True)

        self.window_layer = nn.Linear(hidden_size, 3 * K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

    def init_hidden(
        self, batch_size: int, device: str
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """ Creates a hidden layer in the NN to accomodate for batch size and hidden size """
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        window_vector = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_vector, kappa

    def one_hot_encoding(self, text: torch.Tensor) -> torch.Tensor:
        """ Encodes a given text tensor into column indexes """
        N = text.shape[0]
        U = text.shape[1]
        encoding = text.new_zeros((N, U, self.vocab_size))
        for i in range(N):
            encoding[i, torch.arange(U), text[i].long()] = 1.0
        return encoding

    def compute_window_vector(
        self, mix_params: torch.Tensor, prev_kappa: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Determines the size of the region for a single pen down to be drawn within. """
        encoding = self.one_hot_encoding(text)
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(text.shape[1], dtype=torch.float32, device=text.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)
        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True
        phi = (phi * text_mask).unsqueeze(2)

        window_vec = torch.sum(phi * encoding, dim=1, keepdim=True)
        return window_vec, prev_kappa

    def forward(
        self,
        inputs: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        initial_hidden: Tuple[torch.Tensor, torch.Tensor],
        prev_window_vec: torch.Tensor,
        prev_kappa: torch.Tensor,
    ) -> Tuple[Any, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """Runs a single pass through the network to determine the probability distribution of the next step (pen up, pen down and vertical position of pen). """

        hid_1_arr = []
        window_vec_arr = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp = torch.cat((inputs[:, t : t + 1, :], prev_window_vec), dim=2)

            hid_1_t, state_1 = self.lstm_1(inp, state_1)
            hid_1_arr.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2), prev_kappa, text, text_mask,
            )

            prev_window_vec = window
            prev_kappa = kappa
            window_vec_arr.append(window)

        hid_1 = torch.cat(hid_1_arr, dim=1)
        window_vec = torch.cat(window_vec_arr, dim=1)

        inp = torch.cat((inputs, hid_1, window_vec), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        # inp = torch.cat((inputs, hid_2), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3], window_vec, prev_kappa
