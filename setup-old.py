# Make a guess as to whether SSE2 is present for now, TODO: Improve
INTEL_LIKE = any(
    val in k.lower()
    for k in platform.uname()
    for val in ("x86", "i686", "i386", "amd64")
)
machine_processor = platform.machine() + platform.processor()
ARM_LIKE = any(machine_processor.startswith(name) for name in ("arm", "aarch"))
if ARM_LIKE:
    print("Processor appears to be ARM")
USE_SSE2 = INTEL_LIKE
NO_SSE2 = os.environ.get("RANDOMGEN_NO_SSE2", "") in (1, "1", "True", "true")
NO_SSE2 = NO_SSE2 or "--no-sse2" in sys.argv
if NO_SSE2:
    USE_SSE2 = False
print(f"Building with SSE?: {USE_SSE2}")



UNDEF_MACROS = []
if os.name == "nt":
    EXTRA_LINK_ARGS = ["/LTCG", "/OPT:REF", "Advapi32.lib", "Kernel32.lib"]
    if DEBUG:
        EXTRA_LINK_ARGS += ["-debug"]
        EXTRA_COMPILE_ARGS += ["-Zi", "/Od"]
        UNDEF_MACROS += ["NDEBUG"]
    if sys.version_info < (3, 0):
        EXTRA_INCLUDE_DIRS += [src_join("common")]
elif DEBUG:
    EXTRA_COMPILE_ARGS += ["-g", "-O0"]
    EXTRA_LINK_ARGS += ["-g"]
    UNDEF_MACROS += ["NDEBUG"]


files = glob.glob("./randomgen/*.in") + glob.glob("./randomgen/legacy/*.in")
for templated_file in files:
    output_file_name = splitext(templated_file)[0]
    with open(templated_file) as source_file:
        template = tempita.Template(source_file.read())
    processed = template.substitute().replace("\r\n", "\n")
    contents = ""
    if exists(output_file_name):
        with open(output_file_name) as output_file:
            contents = output_file.read()
    if contents != processed:
        print(f"Processing {templated_file} to {output_file_name}")
        with open(output_file_name, "w", newline="\n") as output_file:
            output_file.write(processed)
