const form = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitButton = document.getElementById("submit-button");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const chipsEl = document.getElementById("chips");

checkOnlineStatus();

chipsEl.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip) return;
  promptInput.value = chip.dataset.prompt;
  promptInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  resultEl.hidden = true;
  submitButton.disabled = true;
  setStatus(
    "loading",
    "Planning your trip — this makes several real OpenAI calls and can take 30-60 seconds..."
  );

  try {
    const response = await fetch("/trips/plan-from-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
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

async function checkOnlineStatus() {
  try {
    const response = await fetch("/health");
    setOnlineStatus(response.ok);
  } catch {
    setOnlineStatus(false);
  }
}

function setOnlineStatus(isOnline) {
  statusDot.className = `dot ${isOnline ? "online" : "offline"}`;
  statusText.textContent = isOnline ? "Online" : "Offline";
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

// Set from the rendered trip's own `currency`, so a trip stored before the
// switch to rupees still renders in the currency it was actually priced in.
let tripCurrency = "INR";

function money(amount) {
  if (amount === null || amount === undefined) return null;
  // en-IN gives rupees their conventional grouping — 1,50,000 rather than 150,000.
  const locale = tripCurrency === "INR" ? "en-IN" : undefined;
  const symbol = tripCurrency === "INR" ? "₹" : "$";
  return `${symbol}${Number(amount).toLocaleString(locale, { maximumFractionDigits: 0 })}`;
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
  const price = money(flight.total);
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
  const nightly = money(hotel.nightly);
  const total = money(hotel.total);
  const cost = nightly && total ? ` — ${nightly}/night, ${total} total` : "";
  return `
    <li class="${isSelected ? "selected" : ""}">
      <span><span class="hotel-name">${escapeHtml(hotel.name)}</span>${tag}<span class="hotel-meta">${cost}</span></span>
      ${isSelected ? '<span class="selected-tag">Selected</span>' : ""}
    </li>`;
}

function renderActivity(activity) {
  const cost = money(activity.estimated_cost);
  const where = activity.location ? ` @ ${escapeHtml(activity.location)}` : "";
  return `
    <div class="activity">
      <div class="activity-row">
        <span class="activity-time">${escapeHtml(activity.time)}</span>
        <span class="activity-title">${escapeHtml(activity.title)}${where}</span>
        <span class="activity-cost">${cost ? `~${cost}` : ""}</span>
      </div>
      ${activity.description ? `<p class="activity-description">${escapeHtml(activity.description)}</p>` : ""}
    </div>`;
}

function renderDay(day) {
  return `
    <div class="day">
      <h4>Day ${day.day} — ${escapeHtml(day.date)}: ${escapeHtml(day.summary)}</h4>
      ${day.activities.map(renderActivity).join("")}
    </div>`;
}

function renderTrip(trip) {
  const it = trip.itinerary;
  tripCurrency = it.currency || "INR";
  const total = money(it.total_estimated_cost);

  resultEl.hidden = false;
  resultEl.innerHTML = `
    <div class="result-header">
      <h2>Your Trip Plan</h2>
      <span class="trip-id">Trip ID: ${escapeHtml(trip.id)}</span>
    </div>

    <div class="card trip-header">
      <h2>${escapeHtml(it.destination)} — ${escapeHtml(it.start_date)} to ${escapeHtml(it.end_date)}, ${it.travelers} traveller(s)</h2>
      ${total ? `<p class="trip-cost">Estimated total: ${total}</p>` : ""}
      ${trip.summary ? `<p class="summary">${escapeHtml(trip.summary)}</p>` : ""}
    </div>

    <div class="card">
      <h3>Flights</h3>
      ${renderFlightRow("Outbound", it.outbound_flight)}
      ${renderFlightRow("Return", it.return_flight)}
    </div>

    ${
      it.lodging_options.length
        ? `<div class="card">
            <h3>Where to stay</h3>
            <ul class="hotel-list">
              ${it.lodging_options.map((h, i) => renderHotelItem(h, i === 0)).join("")}
            </ul>
          </div>`
        : ""
    }

    <div class="card">
      <h3>Day by day</h3>
      ${it.days.map(renderDay).join("")}
    </div>

    ${
      it.notes.length
        ? `<div class="card notes">
            <h3>Notes</h3>
            <ul>${it.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
          </div>`
        : ""
    }
  `;
}
