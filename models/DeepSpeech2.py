from torch import nn


class DS2LargeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv_layers = nn.Sequential(
            # Padding is based on padding='same'
            nn.Conv2d(in_channels=cfg.audio_channels, out_channels=32, kernel_size=(11, 41), stride=(1, 2),
                      padding=(5, 20)),
            self.add_batch_norm(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            self.add_batch_norm(32),
            nn.ReLU()
        )

        self.rnn_layers = nn.GRU(
            input_size=32 * 32,  # calculated based on convolutions
            hidden_size=800,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
            dropout=1 - self.cfg.dropout_keep_prob
        )

        self.fc = nn.Linear(800 * 2, 800 * 2)
        self.projection = nn.Linear(800 * 2, cfg.n_tokens)

    def add_batch_norm(self, num_channels):
        if self.cfg.batch_norm:
            return nn.BatchNorm2d(num_channels)
        else:
            return nn.Identity()  # Use an Identity layer if batch_norm is False

    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)  # turn x shape to [Batch=16, channels=1, Time=513, n_mels=128]
        x = self.conv_layers(x)

        # Reshape for RNN input to [Batch, Time, width*channels]
        batch_size, num_channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, height, num_channels * width)

        # RNN layers
        x, _ = self.rnn_layers(x)

        # Fully connected layer
        x = self.fc(x)

        x = nn.ReLU()(x)
        x = nn.Dropout(p=1 - self.cfg.dropout_keep_prob)(x)

        # Projection layer
        x = self.projection(x)
        return x


class DS2SmallModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv_layers = nn.Sequential(
            # Padding is based on padding='same'
            nn.Conv2d(in_channels=cfg.audio_channels, out_channels=32, kernel_size=(11, 41), stride=(1, 2),
                      padding=(5, 20)),
            self.add_batch_norm(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            self.add_batch_norm(32),
            nn.ReLU()
        )

        self.rnn_layers = nn.GRU(
            input_size=32 * 32,  # calculated based on convolutions
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=1 - self.cfg.dropout_keep_prob
        )

        self.fc = nn.Linear(512 * 2, 512 * 2)
        self.projection = nn.Linear(512 * 2, cfg.n_tokens)

    def add_batch_norm(self, num_channels):
        if self.cfg.batch_norm:
            return nn.BatchNorm2d(num_channels)
        else:
            return nn.Identity()  # Use an Identity layer if batch_norm is False

    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)  # turn x shape to [Batch=16, channels=1, Time=513, n_mels=128]
        x = self.conv_layers(x)

        # Reshape for RNN input to [Batch, Time, width*channels]
        batch_size, num_channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, height, num_channels * width)

        # RNN layers
        x, _ = self.rnn_layers(x)

        # Fully connected layer
        x = self.fc(x)

        x = nn.ReLU()(x)
        x = nn.Dropout(p=1 - self.cfg.dropout_keep_prob)(x)

        # Projection layer
        x = self.projection(x)
        return x


class DS2ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv_layers = nn.Sequential(
            # Padding is based on padding='same'
            nn.Conv2d(in_channels=cfg.audio_channels, out_channels=32, kernel_size=(11, 41), stride=(1, 2),
                      padding=(5, 20)),
            self.add_batch_norm(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            self.add_batch_norm(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            self.add_batch_norm(96),
            nn.ReLU()
        )

        self.rnn_layers = nn.GRU(
            input_size=96 * 16,  # calculated based on convolutions
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(256 * 2, 256 * 2)
        self.projection = nn.Linear(256 * 2, cfg.n_tokens)

    def add_batch_norm(self, num_channels):
        if self.cfg.batch_norm:
            return nn.BatchNorm2d(num_channels)
        else:
            return nn.Identity()  # Use an Identity layer if batch_norm is False

    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)  # turn x shape to [Batch=16, channels=1, Time=513, n_mels=128]
        x = self.conv_layers(x)

        # Reshape for RNN input to [Batch, Time, width*channels]
        batch_size, num_channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, height, num_channels * width)

        # RNN layers
        x, _ = self.rnn_layers(x)

        # Fully connected layer
        x = self.fc(x)

        x = nn.ReLU()(x)
        x = nn.Dropout(p=1 - self.cfg.dropout_keep_prob)(x)

        # Projection layer
        x = self.projection(x)
        return x
