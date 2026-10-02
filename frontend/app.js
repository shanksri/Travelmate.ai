const form = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const submitButton = document.getElementById("submit-button");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const chipsEl = document.getElementById("chips");
const includeFlights = document.getElementById("include-flights");
const includeHotels = document.getElementById("include-hotels");
const includeRestaurants = document.getElementById("include-restaurants");
const startDateInput = document.getElementById("start-date");
const endDateInput = document.getElementById("end-date");
const reviseCard = document.getElementById("revise-card");
const reviseForm = document.getElementById("revise-form");
const reviseInput = document.getElementById("revise-input");
const reviseButton = document.getElementById("revise-button");

// The trip every change applies to. Each change is saved server-side as the
// next version of this thread, so earlier versions are never overwritten.
let currentThreadId = null;

checkOnlineStatus();

// Same limit as the backend's MAX_TRIP_NIGHTS.
const MAX_TRIP_NIGHTS = 60;

// yyyy-mm-dd in the viewer's own timezone — toISOString() would give the UTC
// date, which is yesterday for anyone in India before 05:30.
function isoDate(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function addDays(iso, days) {
  const d = new Date(`${iso}T00:00:00`);
  d.setDate(d.getDate() + days);
  return isoDate(d);
}

// Keep the calendars from offering impossible picks: nothing in the past, no
// end before the start, nothing past the longest trip allowed.
function updateDateLimits() {
  const today = isoDate(new Date());
  startDateInput.min = today;
  const start = startDateInput.value;
  endDateInput.min = start || today;
  endDateInput.max = start ? addDays(start, MAX_TRIP_NIGHTS) : "";
  if (start && endDateInput.value && (endDateInput.value < endDateInput.min || endDateInput.value > endDateInput.max)) {
    endDateInput.value = "";
  }
}

updateDateLimits();
startDateInput.addEventListener("change", updateDateLimits);

// Returns {start_date, end_date} (both null when neither is picked), or
// throws with a message for the status line.
function pickedDates() {
  const start = startDateInput.value;
  const end = endDateInput.value;
  if (!start && !end) return { start_date: null, end_date: null };
  if (!start || !end) throw new Error("Pick both a start and an end date, or leave both empty.");
  if (start < isoDate(new Date())) throw new Error("The start date is in the past.");
  if (end < start) throw new Error("The end date is before the start date.");
  if (end > addDays(start, MAX_TRIP_NIGHTS)) throw new Error("Trips longer than 60 days aren't supported.");
  return { start_date: start, end_date: end };
}

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

  let dates;
  try {
    dates = pickedDates();
  } catch (err) {
    setStatus("error", err.message);
    return;
  }

  resultEl.hidden = true;
  reviseCard.hidden = true;
  submitButton.disabled = true;
  setStatus("loading", "Starting…");

  try {
    // A job runs on the server and streams what it's doing (app/jobs.py);
    // the sentence decides whether it's a trip, places or a route.
    const response = await fetch("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        include_flights: includeFlights.checked,
        include_hotels: includeHotels.checked,
        include_restaurants: includeRestaurants.checked,
        ...dates,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(data));
    }
    // In the URL, so a reload — or reopening the tab — picks the job back up.
    history.replaceState(null, "", `#job=${data.job_id}`);
    followJob(data.job_id);
  } catch (err) {
    setStatus("error", err.message || "Something went wrong.");
    submitButton.disabled = false;
  }
});

// --- following a job's events ----------------------------------------------------
// The server sends each step as it happens: status lines, flights and hotels
// as they're found, each day the moment it's written, the saved trip, then
// each city's places. `live` collects them; `renderLive` redraws from it.

let jobSource = null;
let live = null;

function jobInUrl() {
  const match = location.hash.match(/job=([A-Za-z0-9]+)/);
  return match ? match[1] : null;
}

function followJob(jobId) {
  if (jobSource) jobSource.close();
  live = { kind: null, flights: null, hotels: null, days: [], trip: null, cities: [], failed: false };
  submitButton.disabled = true;

  jobSource = new EventSource(`/jobs/${encodeURIComponent(jobId)}/events`);
  jobSource.onmessage = (message) => handleJobEvent(JSON.parse(message.data));
  jobSource.onerror = () => {
    // A dropped connection reconnects by itself (and resumes where it left
    // off); CLOSED means the server doesn't know this job — it restarted, or
    // the job is over an hour old.
    if (jobSource && jobSource.readyState === EventSource.CLOSED) {
      jobSource = null;
      submitButton.disabled = false;
      history.replaceState(null, "", location.pathname);
      setStatus(
        "error",
        "This plan's progress isn't available any more (the server may have restarted). " +
          "A trip that had finished is still saved; otherwise, please start again."
      );
    }
  };
}

function handleJobEvent(event) {
  switch (event.type) {
    case "status":
      setStatus("loading", event.message);
      break;
    case "intent":
      live.kind = event.kind;
      break;
    case "flights":
      live.flights = event;
      renderLive();
      break;
    case "hotels":
      live.hotels = event;
      renderLive();
      break;
    case "day":
      live.days.push(event.day);
      renderLive();
      break;
    case "days_reset":
      live.days = [];
      renderLive();
      break;
    case "trip":
      live.trip = event.trip;
      renderLive();
      break;
    case "city":
      live.cities.push(event.city);
      renderLive();
      break;
    case "places":
      renderPlaces(event.places);
      break;
    case "route":
      renderRoute(event.route);
      break;
    case "error":
      live.failed = true;
      setStatus("error", event.detail);
      break;
    case "done":
      jobSource.close();
      jobSource = null;
      if (!live.failed) hideStatus();
      submitButton.disabled = false;
      break;
  }
}

// Cities arrive in whatever order their searches finish; shown in the order
// the trip visits them.
function citiesInTripOrder(trip, cities) {
  const order = new Map();
  for (const day of trip.itinerary.days) {
    const city = (day.city || "").toLowerCase();
    if (city && !order.has(city)) order.set(city, order.size);
  }
  const rank = (c) => order.get(c.city.toLowerCase()) ?? order.size;
  return [...cities].sort((a, b) => rank(a) - rank(b));
}

function renderLive() {
  if (live.trip) {
    renderTrip(live.trip);
    renderCityPlaces(citiesInTripOrder(live.trip, live.cities));
    return;
  }

  // Still planning: show what's arrived so far.
  reviseCard.hidden = true;
  resultEl.hidden = false;
  const flights = live.flights;
  const hotels = live.hotels;
  const writing = live.days.length ? "Writing the next day…" : "Writing your plan…";
  resultEl.innerHTML = `
    <div class="result-header">
      <h2>Your Trip Plan</h2>
      <span class="trip-id">Planning…</span>
    </div>
    ${flights ? renderFlightTable("Outbound options", flights.outbound_options, null, flights.outbound_source) : ""}
    ${flights ? renderFlightTable("Return options", flights.return_options, null, flights.return_source) : ""}
    ${hotels ? renderHotelCard(hotels.options, hotels.source) : ""}
    <div class="card">
      <h3>Day by day</h3>
      ${live.days.map(renderDay).join("")}
      <p class="writing">${writing}</p>
    </div>
  `;
}

reviseForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const changeRequest = reviseInput.value.trim();
  if (!changeRequest || !currentThreadId) return;

  reviseButton.disabled = true;
  submitButton.disabled = true;
  setStatus("loading", "Updating your plan — this rewrites the itinerary and can take 30-60 seconds...");

  try {
    const response = await fetch(`/trips/${encodeURIComponent(currentThreadId)}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ change_request: changeRequest }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(extractErrorMessage(data));
    }

    hideStatus();
    reviseInput.value = "";
    // Newer than what the job in the URL produced, so a reload mustn't
    // bring that back.
    history.replaceState(null, "", location.pathname);
    renderTrip(data.trip);
  } catch (err) {
    // The plan on screen is untouched, so the change can simply be retried.
    setStatus("error", err.message || "Something went wrong.");
  } finally {
    reviseButton.disabled = false;
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

const OPTIONS_PER_ROW = 3;

function flightDuration(hours) {
  if (hours === null || hours === undefined) return "—";
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return m ? `${h}h ${m}m` : `${h}h`;
}

// Read straight off the ISO string rather than through `new Date(...)`, which
// would shift the time into the viewer's timezone — a 06:00 departure from
// Delhi should read 06:00 wherever the page is opened.
function flightWhen(flight) {
  const iso = flight.departure_at || flight.depart_date;
  if (!iso) return "";
  const day = new Date(`${iso.slice(0, 10)}T00:00:00`).toLocaleDateString("en-IN", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  return flight.departure_at ? `${day}, ${iso.slice(11, 16)}` : day;
}

// The booked leg is a copy of one of the options (plus the agent's rationale),
// so it's recognised by its fields rather than by position.
function sameFlight(a, b) {
  if (!a || !b) return false;
  return ["carrier", "depart_date", "departure_at", "stops", "duration_hours", "total"].every(
    (key) => (a[key] ?? null) === (b[key] ?? null)
  );
}

function renderFlightOption(flight, booked) {
  const stops = flight.stops === 0 ? "nonstop" : `${flight.stops} stop(s)`;
  const price = money(flight.total);
  const isBooked = sameFlight(flight, booked);
  return `
    <div class="option-row${isBooked ? " booked" : ""}">
      <span class="option-info">
        <span class="option-duration">${flightDuration(flight.duration_hours)}</span>
        <span class="option-stops">${stops}</span>
        <span class="option-meta">${escapeHtml(flightWhen(flight))}</span>
        ${isBooked ? '<span class="selected-tag">Booked</span>' : ""}
      </span>
      <span class="option-price">${price ?? "—"}</span>
    </div>`;
}

// Cheapest and fastest are picked here rather than server-side so both rows
// come from the one option list the itinerary already carries.
// --- where the data came from --------------------------------------------------
// Every flight and hotel result carries a `source` (app/models/itinerary.py
// DataSource): live, cached, stale, unavailable or sample. Shown as one line,
// so sample or old data can't pass for live prices.

function timeAgo(iso) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  return `${hours} h ago`;
}

function sourceLink(source) {
  return source.link
    ? ` <a href="${escapeHtml(source.link)}" target="_blank" rel="noopener">Search ${escapeHtml(source.provider)}</a>`
    : "";
}

function sourceLine(source) {
  if (!source) return "";
  const provider = escapeHtml(source.provider);
  const when = source.fetched_at ? timeAgo(source.fetched_at) : "";
  const reason = escapeHtml(source.detail || "");
  switch (source.status) {
    case "live":
      return `<p class="source-line">Live from ${provider}</p>`;
    case "cached":
      return `<p class="source-line">From ${provider}, checked ${when}</p>`;
    case "stale":
      return `<p class="source-line warn" title="${reason}">Couldn't refresh just now, so these are ${provider} prices from ${when}.${sourceLink(source)}</p>`;
    case "unavailable":
      return `<p class="source-line warn" title="${reason}">Couldn't get results from ${provider} right now.${sourceLink(source)}</p>`;
    case "sample":
      return '<p class="source-line warn">Sample data, not real prices</p>';
    default:
      return "";
  }
}

// A leg with no options: say why instead of leaving a gap.
function renderNoFlights(label, source) {
  if (!source || source.status === "sample") return "";
  const line =
    source.status === "unavailable"
      ? sourceLine(source)
      : `<p class="source-line">${escapeHtml(source.provider)} has no flights for this route on this date.${sourceLink(source)}</p>`;
  return `<div class="card"><h3>${escapeHtml(label)}</h3>${line}</div>`;
}

function renderFlightTable(label, options, booked, source) {
  if (!options || !options.length) return renderNoFlights(label, source);

  const route = options[0].origin && options[0].destination
    ? ` — ${escapeHtml(options[0].origin)} → ${escapeHtml(options[0].destination)}`
    : "";
  const byPrice = [...options].sort((a, b) => (a.total ?? Infinity) - (b.total ?? Infinity));
  const byDuration = [...options]
    // A missing duration would otherwise sort to the front and be presented
    // as the fastest flight available.
    .filter((f) => f.duration_hours !== null && f.duration_hours !== undefined)
    .sort((a, b) => a.duration_hours - b.duration_hours);

  const row = (title, list) =>
    list.length
      ? `<tr>
           <th scope="row">${title}</th>
           <td colspan="2">${list.slice(0, OPTIONS_PER_ROW).map((f) => renderFlightOption(f, booked)).join("")}</td>
         </tr>`
      : "";

  return `
    <div class="card">
      <h3>${escapeHtml(label)}${route}</h3>
      ${sourceLine(source)}
      <table class="flight-table">
        <thead>
          <tr><th></th><th>Flight</th><th class="price-col">Price</th></tr>
        </thead>
        <tbody>
          ${row("Cheapest", byPrice)}
          ${row("Fastest", byDuration)}
        </tbody>
      </table>
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
  const where = activity.location ? ` @ ${escapeHtml(activity.location)}` : "";
  return `
    <div class="activity">
      <div class="activity-row">
        <span class="activity-time">${escapeHtml(activity.time)}</span>
        <span class="activity-title">${escapeHtml(activity.title)}${where}</span>
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

// --- Google Maps answers ------------------------------------------------------
// Google's terms: every result shows its attribution, and results aren't
// stored — so these are rendered and forgotten, and the revise box (which
// only applies to saved trips) stays hidden.

// Each attribution's title is "<place name> - Google Maps"; shown as one
// "From Google Maps:" line with every source linked.
function renderAttribution(attributions) {
  const seen = new Map();
  for (const a of attributions) {
    if (!a || !a.title) continue;
    const name = a.title.replace(/\s*-\s*Google Maps\s*$/, "");
    if (!seen.has(name)) seen.set(name, a.url);
  }
  const links = [...seen].map(([name, url]) =>
    url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(name)}</a>` : escapeHtml(name)
  );
  return `<p class="maps-attribution">From Google Maps${links.length ? `: ${links.join(" · ")}` : ""}</p>`;
}

// The summary cites places as [0], [1]… — each becomes a numbered link to that
// place on Google Maps. **bold** is the one bit of markdown it uses.
function renderSummary(summary, places) {
  const byIndex = new Map(places.map((p) => [p.index, p]));
  return escapeHtml(summary)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, (match, n) => {
      const place = byIndex.get(Number(n));
      if (!place || !place.place_url) return "";
      return `<a class="place-cite" href="${escapeHtml(place.place_url)}" target="_blank" rel="noopener" title="Open in Google Maps">${Number(n) + 1}</a>`;
    })
    .replace(/\n/g, "<br>");
}

function showMapsResult(html) {
  reviseCard.hidden = true;
  currentThreadId = null;
  resultEl.hidden = false;
  resultEl.innerHTML = html;
}

function placesCard(places) {
  return `
    <div class="card">
      <h3>${escapeHtml(places.query)}</h3>
      <p class="places-summary">${renderSummary(places.summary, places.places)}</p>
      ${renderAttribution(places.places.map((p) => p.attribution))}
    </div>`;
}

function renderPlaces(places) {
  showMapsResult(placesCard(places));
}

// A trip that asked for places in each city gets one card per city under the
// plan. Not part of the saved trip (Google's terms), so a revised version of
// the plan is shown without them.
function renderCityPlaces(cities) {
  if (!cities || !cities.length) return;
  const cards = cities.map((c) =>
    c.places
      ? placesCard(c.places)
      : `<div class="card"><h3>${escapeHtml(c.city)}</h3><p class="route-warning">Couldn't load places for ${escapeHtml(c.city)}: ${escapeHtml(c.error || "unknown error")}</p></div>`
  );
  resultEl.insertAdjacentHTML("beforeend", cards.join(""));
}

function routeDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const minutes = Math.round(seconds / 60);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return h ? (m ? `${h} h ${m} min` : `${h} h`) : `${m} min`;
}

function renderRoute(route) {
  const km = route.distance_meters === null || route.distance_meters === undefined
    ? "—"
    : `${(route.distance_meters / 1000).toLocaleString("en-IN", { maximumFractionDigits: 0 })} km`;
  const walking = route.travel_mode === "WALK";
  showMapsResult(`
    <div class="card">
      <h3>${escapeHtml(route.origin)} → ${escapeHtml(route.destination)}</h3>
      <p class="route-line"><strong>${km}</strong> · about <strong>${routeDuration(route.duration_seconds)}</strong> ${walking ? "on foot" : "by road"}</p>
      ${walking ? '<p class="route-warning">Walking routes are in beta and may be missing clear sidewalks or pedestrian paths.</p>' : ""}
      <p><a href="${escapeHtml(route.maps_url)}" target="_blank" rel="noopener">Open directions in Google Maps</a></p>
      ${renderAttribution([route.attribution])}
    </div>`);
}

function renderHotelCard(options, source) {
  if (options && options.length) {
    return `<div class="card">
        <h3>Where to stay</h3>
        ${sourceLine(source)}
        <ul class="hotel-list">
          ${options.map((h, i) => renderHotelItem(h, i === 0)).join("")}
        </ul>
      </div>`;
  }
  if (source && source.status === "unavailable") {
    return `<div class="card"><h3>Where to stay</h3>${sourceLine(source)}</div>`;
  }
  return "";
}

function renderTrip(trip) {
  const it = trip.itinerary;
  tripCurrency = it.currency || "INR";
  const total = money(it.total_estimated_cost);

  currentThreadId = trip.thread_id || trip.id;
  reviseCard.hidden = false;

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

    ${renderFlightTable("Outbound options", it.outbound_options, it.outbound_flight, it.outbound_source)}
    ${renderFlightTable("Return options", it.return_options, it.return_flight, it.return_source)}

    ${renderHotelCard(it.lodging_options, it.lodging_source)}

    <div class="card">
      <h3>Day by day</h3>
      ${it.days.map(renderDay).join("")}
    </div>
  `;
}

// A job in the URL — after a reload, or a reopened tab — is picked back up.
const jobToResume = jobInUrl();
if (jobToResume) followJob(jobToResume);
