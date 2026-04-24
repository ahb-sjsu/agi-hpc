"use client";

import {
  ParticipantTile,
  useEnsureTrackRef,
  useMultibandTrackVolume,
  useTracks,
  type TrackReferenceOrPlaceholder,
} from "@livekit/components-react";
import { Track } from "livekit-client";

/**
 * ParticipantTileWithMeter — wraps the default LiveKit tile and
 * overlays a small audio-level meter so each participant has a
 * visible "mic is working" indicator.
 *
 * The default ``<ParticipantTile />`` only shows a speaking ring
 * when the participant is actively above threshold. That gives
 * no feedback when someone's mic is on but they aren't speaking
 * (is my mic working? am I muted?). A real-time band meter
 * answers that unambiguously.
 *
 * The meter renders five bands fed by
 * :func:`useMultibandTrackVolume`. When the track is missing or
 * muted the meter flattens to its floor and a mic-muted pill
 * appears.
 */
export default function ParticipantTileWithMeter() {
  // Inside a <GridLayout>'s child slot, the tile is rendered per
  // track reference — useEnsureTrackRef pulls the one for this tile.
  const trackRef = useEnsureTrackRef();

  return (
    <div className="relative h-full w-full">
      <ParticipantTile />
      <MicMeterOverlay trackRef={trackRef} />
    </div>
  );
}

/**
 * MicMeterOverlay — bottom-left overlay showing the participant's
 * microphone activity. Reads the mic track for the participant that
 * owns this tile; if the tile's own trackRef is already audio it
 * uses that directly.
 */
function MicMeterOverlay({
  trackRef,
}: {
  trackRef: TrackReferenceOrPlaceholder;
}) {
  // The tile is usually instantiated for the camera track; we want
  // to meter the mic of the *same participant*. Pull all mic tracks
  // from the room and pick the one belonging to this tile's
  // participant.
  const micTracks = useTracks([Track.Source.Microphone]);
  const ownMic = micTracks.find(
    (t) => t.participant.identity === trackRef.participant.identity,
  );

  // If we can't find a mic track, render "no mic" state.
  const micMuted = !ownMic || !ownMic.publication || ownMic.publication.isMuted;

  return (
    <div
      aria-live="off"
      className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1.5 rounded bg-bg/60 px-1.5 py-1 backdrop-blur-sm"
      title={micMuted ? "microphone muted" : "microphone active"}
    >
      <MicGlyph muted={micMuted} />
      {ownMic && !micMuted ? (
        <MicBars trackRef={ownMic} />
      ) : (
        <span className="text-[9px] font-mono uppercase tracking-wider text-text-muted">
          muted
        </span>
      )}
    </div>
  );
}

const BAND_COUNT = 5;

function MicBars({ trackRef }: { trackRef: TrackReferenceOrPlaceholder }) {
  // useMultibandTrackVolume polls the audio element and returns an
  // array of per-band peak values in [0, 1]. Five bands is enough
  // for a compact glance-meter; a spectrum analyzer this is not.
  const volumes = useMultibandTrackVolume(trackRef, {
    bands: BAND_COUNT,
    updateInterval: 100,
  });

  return (
    <div
      role="meter"
      aria-label="microphone level"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={Math.max(...volumes, 0)}
      className="flex items-end gap-[2px] h-3"
    >
      {volumes.map((v, i) => {
        // Scale exponentially so quiet speech still shows motion
        // without loud audio saturating immediately.
        const scaled = Math.min(1, Math.pow(v, 0.6));
        const heightPct = Math.max(12, scaled * 100);
        const tone =
          scaled < 0.15 ? "bg-text-muted" : scaled < 0.6 ? "bg-ok" : "bg-warn";
        return (
          <span
            key={i}
            className={`inline-block w-[2px] rounded-sm ${tone}`}
            style={{ height: `${heightPct}%` }}
          />
        );
      })}
    </div>
  );
}

function MicGlyph({ muted }: { muted: boolean }) {
  if (muted) {
    return (
      <svg
        viewBox="0 0 24 24"
        width="12"
        height="12"
        aria-hidden="true"
        className="text-err"
      >
        <path
          fill="currentColor"
          d="M12 2a3 3 0 0 0-3 3v4.7L5.3 6a9 9 0 0 0 2 10.8l1.4-1.4A7 7 0 0 1 6.9 8.3L9 10.4V11a3 3 0 0 0 4.9 2.3l2 2a5 5 0 0 1-2.9.8 5 5 0 0 1-5-5H6a7 7 0 0 0 6 7v2h-3v2h8v-2h-3v-2a7 7 0 0 0 2.6-.7l2.1 2.1L20 17.3 3.7 1zM12 14a3 3 0 0 1-3-3v-.2l3 3z"
        />
      </svg>
    );
  }
  return (
    <svg
      viewBox="0 0 24 24"
      width="12"
      height="12"
      aria-hidden="true"
      className="text-text-dim"
    >
      <path
        fill="currentColor"
        d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zm5 9a5 5 0 0 1-10 0H5a7 7 0 0 0 6 7v2H7v2h10v-2h-4v-2a7 7 0 0 0 6-7z"
      />
    </svg>
  );
}
