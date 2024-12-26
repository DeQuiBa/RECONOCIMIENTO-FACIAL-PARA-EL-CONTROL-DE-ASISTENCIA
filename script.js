// Simulación de un reconocimiento facial
function identificarPersona(nombre) {
    // Supongamos que "Deivis" es la persona identificada
    if (nombre === "Deivis") {
        document.getElementById("persona").innerText = "Identificado: Deivis";
        document.getElementById("nombre").value = "Deivis"; // Autocompletar nombre
        document.getElementById("asistencia").value = "A"; // Autocompletar asistencia con "A"
    } else {
        document.getElementById("persona").innerText = "Persona no reconocida.";
    }
}

// Llamada a la función con "Deivis" como ejemplo
identificarPersona("Deivis");

// Enviar formulario
document.getElementById("formAsistencia").addEventListener("submit", function(event) {
    event.preventDefault();
    const nombre = document.getElementById("nombre").value;
    const asistencia = document.getElementById("asistencia").value;
    alert(`Asistencia registrada para ${nombre}: ${asistencia}`);
});
