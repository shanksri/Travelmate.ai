const form = document.getElementById("trip-form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitButton = document.getElementById("submit-button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = buildPayload(new FormData(form));

  resultEl.hidden = true;
  submitButton.disabled = true;
  setStatus(
    "loading",
    "Planning your trip — this makes several real OpenAI calls and can take 30-60 seconds..."
  );

  try {
    const response = await fetch("/trips/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(extractErrorMessage(data));
    }

    hideStatus();
    renderTrip(data.trip);
  } catch (err) {
    setStatus("error", err.message || "Something went wrong.");
  } finally {
    submitButton.disabled = false;
  }
});

function buildPayload(formData) {
  const payload = {
    start_date: formData.get("start_date"),
    end_date: formData.get("end_date"),
    travelers: Number(formData.get("travelers")) || 1,
    pace: formData.get("pace"),
  };

  const destination = formData.get("destination").trim();
  if (destination) payload.destination = destination;

  const origin = formData.get("origin").trim();
  if (origin) payload.origin = origin;

  const budget = formData.get("budget_usd");
  if (budget) payload.budget_usd = Number(budget);

  const interests = formData.get("interests").trim();
  if (interests) {
    payload.interests = interests
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  const notes = formData.get("notes").trim();
  if (notes) payload.notes = notes;

  return payload;
}

function extractErrorMessage(data) {
  if (!data || data.detail === undefined) return "Request failed.";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    // FastAPI's own 422 validation-error shape: a list of {loc, msg, type}.
    return data.detail
      .map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`)
      .join("; ");
  }
  return "Request failed.";
}

function setStatus(kind, message) {
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`;
  statusEl.textContent = message;
}

function hideStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function money(amount) {
  return amount === null || amount === undefined
    ? null
    : `$${Number(amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function renderFlightRow(label, flight) {
  if (!flight) {
    return `
      <div class="flight-row">
        <span class="flight-label">${label}</span>
        <span class="flight-none">not included</span>
      </div>`;
  }
  const stops = flight.stops === 0 ? "nonstop" : `${flight.stops} stop(s)`;
  const price = money(flight.total_usd);
  return `
    <div class="flight-row">
      <span class="flight-label">${label}</span>
      <span>
        ${escapeHtml(flight.carrier)} — ${escapeHtml(flight.origin)} → ${escapeHtml(flight.destination)},
        ${escapeHtml(flight.depart_date)} (${stops}${price ? `, ${price} total` : ""})
      </span>
    </div>`;
}

function renderHotelItem(hotel, isSelected) {
  const bits = [];
  if (hotel.tier) bits.push(escapeHtml(hotel.tier));
  if (hotel.rating !== null && hotel.rating !== undefined) bits.push(`${hotel.rating} rating`);
  const tag = bits.length ? ` (${bits.join(", ")})` : "";
  const nightly = money(hotel.nightly_usd);
  const total = money(hotel.total_usd);
  const cost = nightly && total ? ` — ${nightly}/night, ${total} total` : "";
  return `
    <li class="${isSelected ? "selected" : ""}">
      <span><span class="hotel-name">${escapeHtml(hotel.name)}</span>${tag}<span class="hotel-meta">${cost}</span></span>
      ${isSelected ? '<span class="selected-tag">Selected</span>' : ""}
    </li>`;
}

function renderActivity(activity) {
  const cost = money(activity.estimated_cost_usd);
  const where = activity.location ? ` @ ${escapeHtml(activity.location)}` : "";
  return `
    <div class="activity">
      <span class="activity-time">${escapeHtml(activity.time)}</span>
      <span>${escapeHtml(activity.title)}${where}</span>
      <span class="activity-cost">${cost ? `~${cost}` : ""}</span>
    </div>`;
}

function renderDay(day) {
  return `
    <div class="day">
      <h3>Day ${day.day} — ${escapeHtml(day.date)}: ${escapeHtml(day.summary)}</h3>
      ${day.activities.map(renderActivity).join("")}
    </div>`;
}

function renderTrip(trip) {
  const it = trip.itinerary;
  const total = money(it.total_estimated_cost);

  resultEl.hidden = false;
  resultEl.innerHTML = `
    <div class="card trip-header">
      <h2>${escapeHtml(it.destination)} — ${escapeHtml(it.start_date)} to ${escapeHtml(it.end_date)}, ${it.travelers} traveller(s)</h2>
      ${total ? `<p class="trip-cost">Estimated total: ${it.currency} ${total}</p>` : ""}
    </div>

    <div class="card">
      <h2>Flights</h2>
      ${renderFlightRow("Outbound", it.outbound_flight)}
      ${renderFlightRow("Return", it.return_flight)}
    </div>

    ${
      it.lodging_options.length
        ? `<div class="card">
            <h2>Where to stay</h2>
            <ul class="hotel-list">
              ${it.lodging_options.map((h, i) => renderHotelItem(h, i === 0)).join("")}
            </ul>
          </div>`
        : ""
    }

    <div class="card">
      <h2>Day by day</h2>
      ${it.days.map(renderDay).join("")}
    </div>

    ${
      it.notes.length
        ? `<div class="card notes">
            <h2>Notes</h2>
            <ul>${it.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
          </div>`
        : ""
    }

    ${
      trip.summary
        ? `<div class="card"><p class="summary">${escapeHtml(trip.summary)}</p></div>`
        : ""
    }
  `;
}
