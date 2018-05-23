import numpy as np
import pytest
import random
from gym_exploration_chain import ExplorationChain


def test_exploration_chain_invalid_init():
    # negative length
    with pytest.raises(ValueError):
        exploration_chain = ExplorationChain(-1)

    # positive length less or equal 3
    with pytest.raises(ValueError):
        exploration_chain = ExplorationChain(3)

    # good length but start state invalid
    with pytest.raises(ValueError):
        exploration_chain = ExplorationChain(4, 5)

    # good length but start state invalid
    with pytest.raises(ValueError):
        exploration_chain = ExplorationChain(4, 4)


def test_exploration_chain_max_steps_working():
    max_steps = 20
    exploration_chain = ExplorationChain(chain_length=10, start_state=1, max_steps=max_steps)
    obs = exploration_chain.reset()
    for i in range(max_steps-1):
        obs, reward, done, info = exploration_chain.step(random.randint(0, 1))
        assert not done
    obs, reward, done, info = exploration_chain.step(random.randint(0, 1))
    assert done


def test_invalid_action():
    max_steps = 20
    exploration_chain = ExplorationChain(chain_length=10, start_state=1, max_steps=max_steps)
    obs = exploration_chain.reset()
    with pytest.raises(ValueError):
        obs, reward, done, info = exploration_chain.step(-1)

    with pytest.raises(ValueError):
        obs, reward, done, info = exploration_chain.step(2)

    with pytest.raises(ValueError):
        obs, reward, done, info = exploration_chain.step(0.5)


def test_reward():
    max_steps = 40
    exploration_chain = ExplorationChain(chain_length=10, start_state=1, max_steps=max_steps)
    obs = exploration_chain.reset()
    for i in range(4):
        _, reward, _, _ = exploration_chain.step(0)
        assert reward == 1/1000
    for i in range(8):
        _, reward, _, _ = exploration_chain.step(1)
        assert reward == 0
    for i in range(5):
        _, reward, _, _ = exploration_chain.step(1)
        assert reward == 1
    for i in range(5):
        _, reward, _, _ = exploration_chain.step(0)
        assert reward == 0


def test_observation_one_hot():
    max_steps = 40
    exploration_chain = ExplorationChain(chain_length=5, start_state=1, max_steps=max_steps,
                                         observation_type=ExplorationChain.ObservationType.OneHot)
    obs = exploration_chain.reset()
    assert np.all(obs == np.array([0, 1, 0, 0, 0]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([0, 0, 1, 0, 0]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([0, 0, 0, 1, 0]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([0, 0, 0, 0, 1]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([0, 0, 0, 0, 1]))

    obs, _, _, _ = exploration_chain.step(0)
    assert np.all(obs == np.array([0, 0, 0, 1, 0]))

    obs = exploration_chain.reset()
    assert np.all(obs == np.array([0, 1, 0, 0, 0]))


def test_observation_therm():
    max_steps = 40
    exploration_chain = ExplorationChain(chain_length=5, start_state=1, max_steps=max_steps,
                                         observation_type=ExplorationChain.ObservationType.Therm)
    obs = exploration_chain.reset()
    assert np.all(obs == np.array([1, 1, 0, 0, 0]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([1, 1, 1, 0, 0]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([1, 1, 1, 1, 0]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([1, 1, 1, 1, 1]))

    obs, _, _, _ = exploration_chain.step(1)
    assert np.all(obs == np.array([1, 1, 1, 1, 1]))

    obs, _, _, _ = exploration_chain.step(0)
    assert np.all(obs == np.array([1, 1, 1, 1, 0]))

    obs = exploration_chain.reset()
    assert np.all(obs == np.array([1, 1, 0, 0, 0]))
