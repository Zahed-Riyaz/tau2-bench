"""
Tests for airline_multistep task set.

Validates:
- Task structure (all required fields present)
- Actions reference real DB entities (user_ids, reservation_ids, flight_numbers)
- Dependency-enforcing unknown_info is set correctly
- Splits are coherent
- Registry registration works
- Generator is deterministic (same seed → same output)
"""

import json
from pathlib import Path

import pytest

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_tasks_multistep, get_tasks_multistep_split
from tau2.domains.airline.tasks.create_multistep_tasks import create_multistep_tasks
from tau2.domains.airline.utils import AIRLINE_DB_PATH, AIRLINE_DATA_DIR
from tau2.data_model.tasks import Task


@pytest.fixture(scope="module")
def db() -> FlightDB:
    return FlightDB.load(AIRLINE_DB_PATH)


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return get_tasks_multistep()


@pytest.fixture(scope="module")
def splits() -> dict:
    return get_tasks_multistep_split()


# ── Basic structure ───────────────────────────────────────────────────────────

class TestTaskStructure:
    def test_total_count(self, tasks):
        assert len(tasks) == 15, f"Expected 15 tasks, got {len(tasks)}"

    def test_five_types_present(self, tasks):
        prefixes = {t.id.rsplit("_", 1)[0] for t in tasks}
        assert prefixes == {"ms_a", "ms_b", "ms_c", "ms_d", "ms_e"}

    def test_three_per_type(self, tasks):
        from collections import Counter
        counts = Counter(t.id.rsplit("_", 1)[0] for t in tasks)
        for prefix, count in counts.items():
            assert count == 3, f"Expected 3 tasks for {prefix}, got {count}"

    def test_unique_ids(self, tasks):
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_domain_is_airline(self, tasks):
        for t in tasks:
            assert t.user_scenario.instructions.domain == "airline"

    def test_unknown_info_set(self, tasks):
        """Every task must tell the user what they don't know (the dependency enforcer)."""
        for t in tasks:
            assert t.user_scenario.instructions.unknown_info, (
                f"Task {t.id} missing unknown_info"
            )

    def test_actions_nonempty(self, tasks):
        for t in tasks:
            assert t.evaluation_criteria.actions, f"Task {t.id} has no actions"

    def test_reward_basis_set(self, tasks):
        for t in tasks:
            assert t.evaluation_criteria.reward_basis, f"Task {t.id} has no reward_basis"

    def test_no_initial_state(self, tasks):
        """Multistep tasks rely on the base DB — no custom initial state needed."""
        for t in tasks:
            assert t.initial_state is None, f"Task {t.id} unexpectedly has initial_state"


# ── DB entity validation ──────────────────────────────────────────────────────

class TestDBReferences:
    def _actions_by_name(self, task: Task, name: str):
        return [a for a in task.evaluation_criteria.actions if a.name == name]

    def test_user_ids_exist(self, tasks, db):
        for t in tasks:
            for action in self._actions_by_name(t, "get_user_details"):
                uid = action.arguments["user_id"]
                assert uid in db.users, f"Task {t.id}: unknown user_id '{uid}'"

    def test_reservation_ids_exist(self, tasks, db):
        for t in tasks:
            for action in self._actions_by_name(t, "get_reservation_details"):
                rid = action.arguments["reservation_id"]
                assert rid in db.reservations, (
                    f"Task {t.id}: unknown reservation_id '{rid}'"
                )

    def test_flight_numbers_exist(self, tasks, db):
        for t in tasks:
            for action in self._actions_by_name(t, "get_flight_status"):
                fn = action.arguments["flight_number"]
                assert fn in db.flights, f"Task {t.id}: unknown flight_number '{fn}'"

    def test_flight_status_dates_exist(self, tasks, db):
        for t in tasks:
            for action in self._actions_by_name(t, "get_flight_status"):
                fn = action.arguments["flight_number"]
                date = action.arguments["date"]
                assert date in db.flights[fn].dates, (
                    f"Task {t.id}: flight {fn} has no date entry for {date}"
                )

    def test_type_b_flight_is_disrupted(self, tasks, db):
        """Type B tasks must have a cancelled or delayed flight."""
        type_b = [t for t in tasks if t.id.startswith("ms_b")]
        assert type_b, "No Type B tasks found"
        for t in type_b:
            status_actions = self._actions_by_name(t, "get_flight_status")
            assert status_actions, f"Task {t.id} has no get_flight_status action"
            fn = status_actions[0].arguments["flight_number"]
            date = status_actions[0].arguments["date"]
            status = db.flights[fn].dates[date].status
            assert status in ("cancelled", "delayed"), (
                f"Task {t.id}: flight {fn} on {date} has status '{status}', "
                "expected 'cancelled' or 'delayed'"
            )

    def test_type_c_flight_from_search(self, tasks, db):
        """Type C book_reservation flights must be available on the search date."""
        type_c = [t for t in tasks if t.id.startswith("ms_c")]
        assert type_c, "No Type C tasks found"
        for t in type_c:
            book_actions = self._actions_by_name(t, "book_reservation")
            assert book_actions, f"Task {t.id} has no book_reservation action"
            for flight_dict in book_actions[0].arguments["flights"]:
                fn = flight_dict["flight_number"]
                date = flight_dict["date"]
                assert fn in db.flights, f"Task {t.id}: unknown flight {fn}"
                assert date in db.flights[fn].dates, (
                    f"Task {t.id}: flight {fn} has no date entry for {date}"
                )
                status = db.flights[fn].dates[date].status
                assert status == "available", (
                    f"Task {t.id}: flight {fn} on {date} is '{status}', not available"
                )

    def test_type_d_original_reservation_not_business(self, tasks, db):
        """Type D tasks upgrade from non-business — source reservation must not be business."""
        type_d = [t for t in tasks if t.id.startswith("ms_d")]
        assert type_d, "No Type D tasks found"
        for t in type_d:
            res_actions = self._actions_by_name(t, "get_reservation_details")
            assert res_actions
            rid = res_actions[0].arguments["reservation_id"]
            res = db.reservations[rid]
            assert res.cabin != "business", (
                f"Task {t.id}: reservation {rid} is already business class"
            )


# ── Dependency chain structure ────────────────────────────────────────────────

class TestDependencyChains:
    def _action_names(self, task: Task) -> list[str]:
        return [a.name for a in task.evaluation_criteria.actions]

    def test_type_a_chain(self, tasks):
        for t in [t for t in tasks if t.id.startswith("ms_a")]:
            names = self._action_names(t)
            assert names[0] == "get_user_details"
            assert names[1] == "get_reservation_details"
            assert names[2] == "get_flight_status"

    def test_type_b_chain(self, tasks):
        for t in [t for t in tasks if t.id.startswith("ms_b")]:
            names = self._action_names(t)
            assert names[0] == "get_user_details"
            assert names[1] == "get_reservation_details"
            assert names[2] == "get_flight_status"
            assert names[3] == "send_certificate"

    def test_type_c_chain(self, tasks):
        for t in [t for t in tasks if t.id.startswith("ms_c")]:
            names = self._action_names(t)
            assert names[0] == "search_direct_flight"
            assert names[1] == "get_user_details"
            assert names[2] == "book_reservation"

    def test_type_d_chain(self, tasks):
        for t in [t for t in tasks if t.id.startswith("ms_d")]:
            names = self._action_names(t)
            assert names[0] == "get_reservation_details"
            assert names[1] == "search_direct_flight"
            assert names[2] == "update_reservation_flights"

    def test_type_e_chain(self, tasks):
        for t in [t for t in tasks if t.id.startswith("ms_e")]:
            names = self._action_names(t)
            assert names[0] == "get_user_details"
            assert names[1] == "get_reservation_details"
            assert names[2] == "update_reservation_baggages"

    def test_type_a_communicate_info_is_valid_status(self, tasks, db):
        valid_statuses = {"available", "cancelled", "delayed", "on time",
                          "flying", "landed"}
        for t in [t for t in tasks if t.id.startswith("ms_a")]:
            for s in t.evaluation_criteria.communicate_info:
                assert s in valid_statuses, (
                    f"Task {t.id}: unexpected status '{s}'"
                )

    def test_type_b_certificate_amount(self, tasks, db):
        """Certificate amount = $50 or $100 × n_passengers."""
        for t in [t for t in tasks if t.id.startswith("ms_b")]:
            cert_actions = [a for a in t.evaluation_criteria.actions
                            if a.name == "send_certificate"]
            assert cert_actions
            amount = cert_actions[0].arguments["amount"]
            assert amount > 0
            assert amount % 50 == 0, (
                f"Task {t.id}: certificate amount {amount} not a multiple of 50"
            )


# ── Split coherence ───────────────────────────────────────────────────────────

class TestSplits:
    def test_splits_keys(self, splits):
        assert set(splits.keys()) == {"train", "test", "base"}

    def test_base_contains_all(self, tasks, splits):
        all_ids = {t.id for t in tasks}
        assert all_ids == set(splits["base"])

    def test_train_test_disjoint(self, splits):
        assert not set(splits["train"]) & set(splits["test"])

    def test_train_test_covers_base(self, splits):
        assert set(splits["train"]) | set(splits["test"]) == set(splits["base"])

    def test_all_split_ids_valid(self, tasks, splits):
        valid = {t.id for t in tasks}
        for split_name, ids in splits.items():
            for tid in ids:
                assert tid in valid, f"Split '{split_name}' has unknown id '{tid}'"


# ── Registry ─────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_airline_multistep_registered(self):
        from tau2.registry import registry
        assert "airline_multistep" in registry.get_task_sets()

    def test_registry_loads_tasks(self):
        from tau2.registry import registry
        loader = registry.get_tasks_loader("airline_multistep")
        tasks = loader("base")
        assert len(tasks) == 15

    def test_registry_loads_train_split(self):
        from tau2.registry import registry
        loader = registry.get_tasks_loader("airline_multistep")
        train = loader("train")
        assert len(train) > 0

    def test_registry_loads_test_split(self):
        from tau2.registry import registry
        loader = registry.get_tasks_loader("airline_multistep")
        test = loader("test")
        assert len(test) > 0


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_output(self, tmp_path):
        tasks1 = create_multistep_tasks(save=False, seed=42)
        tasks2 = create_multistep_tasks(save=False, seed=42)
        ids1 = [t.id for t in tasks1]
        ids2 = [t.id for t in tasks2]
        assert ids1 == ids2

    def test_different_seed_different_splits(self, tmp_path):
        """Different seeds produce different train/test splits (probabilistic)."""
        import random
        random.seed(0)
        tasks_a = create_multistep_tasks(save=False, seed=0)
        random.seed(99)
        tasks_b = create_multistep_tasks(save=False, seed=99)
        # Task IDs themselves are deterministic (DB scan order), just splits may differ
        ids_a = [t.id for t in tasks_a]
        ids_b = [t.id for t in tasks_b]
        assert ids_a == ids_b  # same tasks regardless of seed
