/**
 * Wire-format types for the Halyard Table runtime.
 *
 * These mirror the Python-side Pydantic/JSON-Schema definitions at
 * ``src/agi/halyard/state/schema/character_sheet.schema.json`` and
 * the NATS subject contracts in ``docs/HALYARD_TABLE.md`` §5.
 *
 * Hand-written rather than generated for Sprint 4: a later sprint
 * may swap this for a build-time step that produces TS types
 * directly from the JSON Schema, but for the scaffold we care
 * more about clarity than round-trip guarantees.
 */

// ─────────────────────────────────────────────────────────────────
// Character sheet
// ─────────────────────────────────────────────────────────────────

export type Role =
  | "expedition_lead"
  | "ships_engineer"
  | "medical_officer"
  | "surface_ops_eva"
  | "radiological_chemist"
  | "xenoarchaeology_consultant"
  | "security_officer";

export type Chassis =
  | "baseline_human"
  | "designed"
  | "reverse_cylinder"
  | "blank"
  | "reconstructed"
  | "long_agent";

export type FactionLoyalty =
  | "clean"
  | "chamber"
  | "hollow_hand"
  | "starry_wisdom"
  | "unn_signals"
  | "mi_go_long_agent"
  | "protogen";

export type BondStatus =
  | "intact"
  | "strained"
  | "lost"
  | "grieved"
  | "reaffirmed";

export type ConditionCode =
  | "major_wound"
  | "bout_of_madness"
  | "temporary_insanity"
  | "indefinite_insanity"
  | "underlying_insanity"
  | "phobia"
  | "mania"
  | "drugged"
  | "unconscious"
  | "dying"
  | "suit_damaged"
  | "radiation_dose"
  | "sleep_debt";

export interface Identity {
  name: string;
  age: number;
  origin: string;
  role: Role;
  chassis: Chassis;
  credit_rating: number;
  pronouns?: string;
  voice?: string;
  visual?: string;
}

export interface Characteristics {
  str: number;
  con: number;
  siz: number;
  dex: number;
  app: number;
  int: number;
  pow: number;
  edu: number;
}

export interface Derived {
  hp_max: number;
  mp_max: number;
  san_starting: number;
  san_max: number;
  luck_max: number;
  move: number;
  build: number;
  damage_bonus: string;
  dodge_base: number;
}

export interface Skill {
  value: number;
  improvement_check?: boolean;
  base?: number;
}

export interface Bond {
  id: string;
  tier: 1 | 2 | 3;
  name: string;
  detail?: string;
  status: BondStatus;
}

export interface Condition {
  code: ConditionCode;
  note: string;
  applied_at?: number;
  expires_at?: number;
}

export interface Status {
  hp_current: number;
  mp_current: number;
  san_current: number;
  luck_current: number;
  conditions?: Condition[];
}

export interface EquipmentItem {
  id: string;
  name: string;
  notes?: string;
  qty?: number;
}

export interface CampaignExt {
  faction_loyalty?: FactionLoyalty;
  climactic_vote_direction?: string;
  personal_hook?: string;
  keeper_hook?: string; // never shown to non-keepers
  why_this_contract?: string;
  rem_log_cSv?: number;
}

export interface CharacterSheet {
  schema_version: "1.0";
  session_id: string;
  pc_id: string;
  identity: Identity;
  characteristics: Characteristics;
  derived: Derived;
  skills: Record<string, Skill>;
  bonds: Bond[];
  status: Status;
  equipment?: EquipmentItem[];
  campaign: CampaignExt;
}

// ─────────────────────────────────────────────────────────────────
// DataChannel envelopes — what lands on the LiveKit DataChannel.
// ─────────────────────────────────────────────────────────────────

export type Envelope =
  | ArtemisSay
  | Sigma4Say
  | SceneTrigger
  | DiceRoll;

export interface ArtemisSay {
  kind: "artemis.say";
  text: string;
  turn_id?: string;
  proof_hash?: string;
  ts: number;
}

export interface Sigma4Say {
  kind: "sigma4.say";
  text: string;
  turn_id?: string;
  proof_hash?: string;
  ts: number;
}

export interface SceneTrigger {
  kind: "scene.trigger";
  scene_id: string;
  note?: string;
  ts: number;
}

export interface DiceRoll {
  kind: "dice.roll";
  author: "keeper" | string;
  expr: string;
  result: number;
  ts: number;
}

// ─────────────────────────────────────────────────────────────────
// halyard-state WS events — what lands on
// ws://…/ws/sheets/<session_id>.
// ─────────────────────────────────────────────────────────────────

export type StateEvent = StateHello | SheetUpdate;

export interface StateHello {
  kind: "session.hello";
  session_id: string;
  pc_ids: string[];
}

export interface SheetUpdate {
  kind: "sheet.update";
  pc_id: string;
  sheet: CharacterSheet;
}

// ─────────────────────────────────────────────────────────────────
// Patch envelope sent to halyard-state REST API.
// ─────────────────────────────────────────────────────────────────

export type Author = "keeper" | "player" | "system";

export interface PatchOp {
  op: "add" | "remove" | "replace";
  path: string;
  value?: unknown;
}

export interface PatchEnvelope {
  author: Author;
  author_pc_id?: string;
  patch: PatchOp[];
  reason?: string;
}
