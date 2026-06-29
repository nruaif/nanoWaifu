import itertools
import os
import tempfile
import unittest

import numpy as np
import torch

from adversarial_flow import (
    AdversarialFlowDiscriminator,
    GradientNormalization,
    discriminator_losses,
    generator_losses,
    interpolate_flow,
    sample_adversarial_flow,
    sample_afm_timesteps,
    set_requires_grad,
)
from model_dit import TokenformerDiT
from train import (
    _linear_sum_assignment,
    load_checkpoint,
    sample_training_timesteps,
    save_checkpoint,
    x0_loss_weight,
)


class AssignmentTests(unittest.TestCase):
    def test_hungarian_assignment_is_globally_optimal(self):
        rng = np.random.default_rng(7)
        cost = rng.normal(size=(5, 5)) ** 2
        assignment = _linear_sum_assignment(cost)
        actual = sum(cost[row, column] for row, column in enumerate(assignment))
        expected = min(
            sum(cost[row, column] for row, column in enumerate(columns))
            for columns in itertools.permutations(range(len(cost)))
        )
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(len(set(assignment.tolist())), len(cost))


class TimestepTests(unittest.TestCase):
    def test_flow_matching_timesteps_are_float32_and_bounded(self):
        timesteps = sample_training_timesteps(
            10000,
            torch.device("cpu"),
            minimum=1e-3,
        )
        self.assertEqual(timesteps.dtype, torch.float32)
        self.assertGreaterEqual(timesteps.min().item(), 1e-3)
        self.assertLessEqual(timesteps.max().item(), 1.0 - 1e-3)
        self.assertLessEqual(x0_loss_weight(timesteps, 5.0).max().item(), 5.0)

    def test_afm_designated_transitions(self):
        source, target = sample_afm_timesteps(128, 4, torch.device("cpu"))
        self.assertTrue(torch.allclose(source - target, torch.full_like(source, 0.25)))
        self.assertGreaterEqual(target.min().item(), 0.0)
        self.assertLessEqual(source.max().item(), 1.0)

    def test_afm_sampler_uses_designated_steps(self):
        model = TokenformerDiT(
            in_channels=4,
            dim=32,
            depth=1,
            num_heads=4,
            num_classes=10,
        ).train()

        class Tags:
            @staticmethod
            def process_prompts(prompts, device):
                size = len(prompts)
                return (
                    torch.arange(size, device=device),
                    torch.arange(size, device=device),
                )

        samples = sample_adversarial_flow(
            model,
            Tags(),
            latent_size=2,
            batch_size=2,
            prompts=["a", "b"],
            device=torch.device("cpu"),
            steps=1,
        )
        self.assertEqual(tuple(samples.shape), (2, 4, 2, 2))
        self.assertTrue(model.training)


class AdversarialLossTests(unittest.TestCase):
    def test_losses_are_finite_and_differentiable(self):
        batch_size = 4
        gp_batch = 2
        logits = [
            torch.randn(size, requires_grad=True)
            for size in (batch_size, batch_size, gp_batch, gp_batch)
        ]
        dis = discriminator_losses(
            *logits,
            weighting=torch.ones(batch_size),
            gp_scale=0.25,
            gp_eps=0.01,
            center_scale=0.01,
        )
        self.assertTrue(torch.isfinite(dis["total"]))
        dis["total"].backward()
        self.assertTrue(all(item.grad is not None for item in logits))

        predicted = torch.randn(4, 3, 2, 2, requires_grad=True)
        source = torch.randn_like(predicted)
        logits_real = torch.randn(4)
        logits_fake = predicted.mean(dim=(1, 2, 3))
        gen = generator_losses(
            logits_real,
            logits_fake,
            predicted,
            source,
            weighting=torch.ones(4),
            ot_scale=0.2,
        )
        self.assertTrue(torch.isfinite(gen["total"]))
        gen["total"].backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_gradient_normalization_tracks_finite_scale(self):
        normalizer = GradientNormalization(ema_decay=0.0)
        inputs = torch.randn(3, 4, requires_grad=True)
        normalizer(inputs).sum().backward()
        self.assertGreater(normalizer.square_avg.item(), 0.0)
        self.assertTrue(torch.isfinite(inputs.grad).all())


class DiscriminatorTests(unittest.TestCase):
    def test_discriminator_accepts_relative_batches(self):
        model = AdversarialFlowDiscriminator(
            in_channels=4,
            dim=32,
            depth=2,
            num_heads=4,
            num_classes=10,
        )
        inputs = torch.randn(6, 4, 2, 2)
        y_indices = torch.tensor([1, 2, 3, 4])
        y_offsets = torch.tensor([0, 2])
        timesteps = torch.tensor([0.0, 0.5])
        logits = model(
            inputs,
            y_indices,
            y_offsets,
            timesteps,
            condition_repeats=(2, 2, 1, 1),
        )
        self.assertEqual(tuple(logits.shape), (6,))
        logits.square().mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_alternating_updates_isolate_parameter_gradients(self):
        generator = TokenformerDiT(
            in_channels=4,
            dim=32,
            depth=2,
            num_heads=4,
            num_classes=10,
        )
        discriminator = AdversarialFlowDiscriminator(
            in_channels=4,
            dim=32,
            depth=2,
            num_heads=4,
            num_classes=10,
        )
        data = torch.randn(2, 4, 2, 2)
        y_indices = torch.tensor([1, 2, 3, 4])
        y_offsets = torch.tensor([0, 2])
        source_t, target_t = sample_afm_timesteps(
            2,
            1,
            torch.device("cpu"),
        )
        source = interpolate_flow(data, torch.randn_like(data), source_t)
        target = interpolate_flow(data, torch.randn_like(data), target_t)

        set_requires_grad(generator, False)
        set_requires_grad(discriminator, True)
        with torch.no_grad():
            predicted = generator(
                source,
                source_t,
                y_indices,
                y_offsets,
            )
        real_logits, fake_logits = discriminator(
            torch.cat([target, predicted]),
            y_indices,
            y_offsets,
            target_t,
            condition_repeats=(2, 2),
        ).split(2)
        torch.nn.functional.softplus(
            -(real_logits - fake_logits)
        ).mean().backward()
        self.assertFalse(
            any(parameter.grad is not None for parameter in generator.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in discriminator.parameters()
            )
        )

        discriminator.zero_grad(set_to_none=True)
        set_requires_grad(generator, True)
        set_requires_grad(discriminator, False)
        predicted = generator(source, source_t, y_indices, y_offsets)
        fake_logits = discriminator(
            predicted,
            y_indices,
            y_offsets,
            target_t,
        )
        torch.nn.functional.softplus(-fake_logits).mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in generator.parameters())
        )
        self.assertFalse(
            any(
                parameter.grad is not None
                for parameter in discriminator.parameters()
            )
        )


class CheckpointTests(unittest.TestCase):
    def test_afm_checkpoint_round_trip(self):
        generator = TokenformerDiT(
            in_channels=4,
            dim=32,
            depth=1,
            num_heads=4,
            num_classes=10,
        )
        discriminator = AdversarialFlowDiscriminator(
            in_channels=4,
            dim=32,
            depth=1,
            num_heads=4,
            num_classes=10,
        )
        gen_optimizer = torch.optim.AdamW(generator.parameters())
        dis_optimizer = torch.optim.AdamW(discriminator.parameters())
        normalizer = GradientNormalization()

        with tempfile.TemporaryDirectory() as output_dir:
            save_checkpoint(
                generator,
                gen_optimizer,
                rank=0,
                output_dir=output_dir,
                step=12,
                config={"training": {"max_checkpoints": 1}},
                discriminator=discriminator,
                discriminator_optimizer=dis_optimizer,
                gradient_normalizer=normalizer,
                training_mode="adversarial_flow",
            )
            path = os.path.join(output_dir, "ckpt_step_12.pth")
            checkpoint = load_checkpoint(path, map_location="cpu")

        self.assertEqual(checkpoint["training_mode"], "adversarial_flow")
        self.assertEqual(checkpoint["global_step"], 12)
        self.assertIn("discriminator_state_dict", checkpoint)
        self.assertIn("discriminator_optimizer_state_dict", checkpoint)
        self.assertIn("gradient_normalizer_state_dict", checkpoint)


if __name__ == "__main__":
    unittest.main()
