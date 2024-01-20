use clap::Parser;
use polkavm_common::program::{Opcode, ProgramBlob, Instruction};
use std::collections::HashMap;
use std::{
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum DisassemblyFormat {
    Guest,
    GuestAndNative,
    Native,
    DiffFriendly,
}

#[derive(Parser, Debug)]
#[clap(version)]
enum Args {
    /// Links a given ELF file into a `.polkavm` program blob.
    Link {
        /// The output file.
        #[clap(short = 'o', long)]
        output: PathBuf,

        #[clap(short = 's', long)]
        strip: bool,

        /// Will only run if the output file doesn't exist, or the input is newer.
        #[clap(long)]
        run_only_if_newer: bool,

        /// The input file.
        input: PathBuf,
    },

    /// Disassembles a .polkavm blob into its human-readable assembly.
    Disassemble {
        /// The output file.
        #[clap(short = 'o', long)]
        output: Option<PathBuf>,

        #[clap(short = 'f', long, value_enum, default_value_t = DisassemblyFormat::Guest)]
        format: DisassemblyFormat,

        /// The input file.
        input: PathBuf,
    },

    /// Calculates various statistics for given program blobs.
    Stats {
        /// The input files.
        inputs: Vec<PathBuf>,
    },
}

macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(format!($($arg)*))
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let result = match args {
        Args::Link {
            output,
            input,
            strip,
            run_only_if_newer,
        } => main_link(input, output, strip, run_only_if_newer),
        Args::Disassemble { output, format, input } => main_disassemble(input, format, output),
        Args::Stats { inputs } => main_stats(inputs),
    };

    if let Err(error) = result {
        eprintln!("ERROR: {}", error);
        std::process::exit(1);
    }
}

fn main_link(input: PathBuf, output: PathBuf, strip: bool, run_only_if_newer: bool) -> Result<(), String> {
    if run_only_if_newer {
        if let Ok(output_mtime) = std::fs::metadata(&output).and_then(|m| m.modified()) {
            if let Ok(input_mtime) = std::fs::metadata(&input).and_then(|m| m.modified()) {
                if output_mtime >= input_mtime {
                    return Ok(());
                }
            }
        }
    }

    let mut config = polkavm_linker::Config::default();
    config.set_strip(strip);

    let data = match std::fs::read(&input) {
        Ok(data) => data,
        Err(error) => {
            bail!("failed to read {input:?}: {error}");
        }
    };

    let blob = match polkavm_linker::program_from_elf(config, &data) {
        Ok(blob) => blob,
        Err(error) => {
            bail!("failed to link {input:?}: {error}");
        }
    };

    if let Err(error) = std::fs::write(&output, blob.as_bytes()) {
        bail!("failed to write the program blob to {output:?}: {error}");
    }

    Ok(())
}

fn load_blob(input: &Path) -> Result<ProgramBlob<'static>, String> {
    let data = match std::fs::read(input) {
        Ok(data) => data,
        Err(error) => {
            bail!("failed to read {input:?}: {error}");
        }
    };

    let blob = match polkavm_linker::ProgramBlob::parse(&data[..]) {
        Ok(blob) => blob,
        Err(error) => {
            bail!("failed to parse {input:?}: {error}");
        }
    };

    Ok(blob.into_owned())
}

fn main_stats(inputs: Vec<PathBuf>) -> Result<(), String> {
    let mut map = HashMap::new();
    for opcode in 0..=255 {
        if let Some(opcode) = Opcode::from_u8(opcode) {
            map.insert(opcode, 0);
        }
    }

    for input in inputs {
        let blob = load_blob(&input)?;
        for instruction in blob.instructions() {
            let instruction = match instruction {
                Ok(instruction) => instruction,
                Err(error) => {
                    bail!("failed to parse instruction: {error}");
                }
            };

            *map.get_mut(&instruction.opcode()).unwrap() += 1;
        }
    }

    let mut list: Vec<_> = map.into_iter().collect();
    list.sort_by_key(|(_, count)| core::cmp::Reverse(*count));

    println!("Instruction distribution:");
    for (opcode, count) in list {
        println!("{opcode:>40}: {count}", opcode = format!("{:?}", opcode));
    }

    Ok(())
}

fn main_disassemble(input: PathBuf, format: DisassemblyFormat, output: Option<PathBuf>) -> Result<(), String> {
    let blob = load_blob(&input)?;

    let native = if matches!(format, DisassemblyFormat::Native | DisassemblyFormat::GuestAndNative) {
        if !cfg!(target_arch = "x86_64") {
            bail!("the selected disassembly format is not supported on this architecture");
        }

        let config = match polkavm::Config::from_env() {
            Ok(config) => config,
            Err(error) => bail!("failed to fetch VM configuration from the environment: {error}"),
        };

        let engine = match polkavm::Engine::new(&config) {
            Ok(engine) => engine,
            Err(error) => bail!("failed to create VM engine: {error}"),
        };

        let module = match polkavm::Module::from_blob(&engine, &Default::default(), &blob) {
            Ok(module) => module,
            Err(error) => bail!("failed to compile {input:?}: {error}"),
        };

        let code = match module.machine_code() {
            Some(code) => code.into_owned(),
            None => bail!("currently selected VM backend doesn't provide raw machine code"),
        };

        let instruction_map = match module.nth_instruction_to_code_offset_map() {
            Some(map) => map.to_vec(),
            None => bail!("currently selected VM backend doesn't provide a machine code map"),
        };

        let code_origin = module.machine_code_origin().unwrap_or(0);
        Some((code_origin, code, instruction_map))
    } else {
        None
    };

    let mut current_byte_offset: usize = 0;
    let mut byte_offsets: Vec<usize> = Vec::new();

    for (nth_instruction, maybe_instruction) in blob.instructions().enumerate() {
            let instruction = maybe_instruction.map_err(|error| format!("failed to parse instruction #{nth_instruction}: {error}"))?;
            let (_, instruction_size) = serialize_instruction(&instruction);

            byte_offsets.push(current_byte_offset);
            current_byte_offset += instruction_size;
        }

     match output {
        Some(output) => {
            let fp = match std::fs::File::create(&output) {
                Ok(fp) => fp,
                Err(error) => {
                    bail!("failed to create output file {output:?}: {error}");
                }
            };

            disassemble_into(format, &blob, native, byte_offsets, std::io::BufWriter::new(fp))
        }
        None => {
            let stdout = std::io::stdout();
            disassemble_into(format, &blob, native, byte_offsets, std::io::BufWriter::new(stdout))
        }
    }
}

#[derive(Default)]
struct AssemblyFormatter {
    buffer: String,
}

impl AssemblyFormatter {
    fn emit(
        &mut self,
        indent: bool,
        code_origin: u64,
        mut code: &[u8],
        mut position: usize,
        writer: &mut impl Write,
    ) -> Result<(), std::io::Error> {
        use iced_x86::Formatter;

        let mut formatter = iced_x86::NasmFormatter::new();
        formatter.options_mut().set_space_after_operand_separator(true);
        formatter.options_mut().set_hex_prefix("0x");
        formatter.options_mut().set_hex_suffix("");
        formatter.options_mut().set_uppercase_hex(false);
        formatter.options_mut().set_small_hex_numbers_in_decimal(false);
        formatter.options_mut().set_show_useless_prefixes(true);
        formatter.options_mut().set_branch_leading_zeros(false);
        formatter.options_mut().set_rip_relative_addresses(true);

        loop {
            let mut decoder = iced_x86::Decoder::with_ip(64, code, code_origin, iced_x86::DecoderOptions::NONE);
            if !decoder.can_decode() {
                break;
            }
            let mut instruction = iced_x86::Instruction::default();
            decoder.decode_out(&mut instruction);

            if indent {
                write!(writer, "                                       ")?;
            }
            write!(writer, "{:8x}: ", position as u64 + code_origin)?;

            let start_index = (instruction.ip() - code_origin) as usize;
            let instr_bytes = &code[start_index..start_index + instruction.len()];
            let mut count = 0;
            for b in instr_bytes.iter() {
                write!(writer, "{:02x} ", b)?;
                count += 3;
            }
            while count < 34 {
                write!(writer, " ")?;
                count += 1;
            }

            self.buffer.clear();
            formatter.format(&instruction, &mut self.buffer);
            write!(writer, "{}", self.buffer)?;
            writeln!(writer)?;

            code = &code[instruction.len()..];
            position += instruction.len();
        }

        Ok(())
    }
}

fn serialize_instruction(instruction: &Instruction) -> (String, usize) {
    let mut buffer = [0u8; 16]; // maximum instruction size?
    let size = instruction.serialize_into(&mut buffer);

    let hex_representation = buffer[..size].iter()
        .map(|byte| format!("{:02x}", byte))
        .collect::<Vec<String>>()
        .join(" ");

    (hex_representation, size)
}

fn find_first_instruction_binary(blob: &ProgramBlob) -> Result<Vec<u8>, String> {
    let first_instruction = blob.instructions().next()
        .ok_or("No instructions in the blob")?
        .map_err(|e| format!("Error parsing first instruction: {}", e))?;

    let (binary, _) = serialize_instruction(&first_instruction);

    // Convert the binary string to a byte array
    let mut result = Vec::new();
    for byte_str in binary.split_whitespace() {
        let byte = u8::from_str_radix(byte_str, 16)
            .map_err(|e| format!("Error parsing byte: {}", e))?;
        result.push(byte);
    }

    Ok(result)
}

// Function to find the true byte offset of the first instruction in the blob
fn find_true_byte_offset(blob: &ProgramBlob, first_instruction_binary: &[u8]) -> Result<usize, String> {
    let blob_bytes = blob.as_bytes();

    blob_bytes.windows(first_instruction_binary.len())
        .position(|window| window == first_instruction_binary)
        .ok_or_else(|| "Binary sequence not found".to_string())
}

// Updated calculate_program_base_address function
fn calculate_program_base_address(blob: &ProgramBlob) -> Result<usize, String> {
    let first_instruction_binary = find_first_instruction_binary(blob)?;
    println!("First instruction binary: {:?}", first_instruction_binary);
    find_true_byte_offset(blob, &first_instruction_binary)

}

fn disassemble_into(
    format: DisassemblyFormat,
    blob: &polkavm_linker::ProgramBlob,
    native: Option<(u64, Vec<u8>, Vec<u32>)>,
    byte_offsets: Vec<usize>,
    mut writer: impl Write,
) -> Result<(), String> {
    let mut instructions = Vec::new();
    for (nth_instruction, maybe_instruction) in blob.instructions().enumerate() {
        let instruction = match maybe_instruction {
            Ok(instruction) => instruction,
            Err(error) => {
                bail!("failed to parse instruction #{nth_instruction}: {error}");
            }
        };

        instructions.push(instruction);
    }

    let mut exports_for_jump_target = HashMap::new();
    for (nth_export, export) in blob.exports().enumerate() {
        let export = match export {
            Ok(export) => export,
            Err(error) => {
                bail!("failed to parse instruction export: {error}");
            }
        };

        exports_for_jump_target
            .entry(export.address())
            .or_insert_with(Vec::new)
            .push((nth_export, export));
    }

    let mut jump_table_map = HashMap::new();
    let mut jump_table = Vec::new();
    for maybe_target in blob.jump_table() {
        let target = match maybe_target {
            Ok(target) => target,
            Err(error) => {
                bail!("failed to parse the jump table: {error}");
            }
        };

        let jump_table_index = jump_table.len() + 1;
        jump_table.push(target);
        assert!(jump_table_map.insert(target, jump_table_index).is_none());
    }

    let format_jump_target = |jump_target_counter: u32| {
        if let Some(jump_table_index) = jump_table_map.get(&jump_target_counter) {
            if !matches!(format, DisassemblyFormat::DiffFriendly) {
                format!("@{jump_target_counter:x}: [@dyn {jump_table_index:x}]")
            } else {
                "@_: [_]".to_string()
            }
        } else if !matches!(format, DisassemblyFormat::DiffFriendly) {
            format!("@{jump_target_counter:x}:")
        } else {
            "@_:".to_string()
        }
    };

    let mut fmt = AssemblyFormatter::default();
    let mut last_line_program_entry = None;
    let mut last_full_name = String::new();
    let mut jump_target_counter = 0;
    let mut pending_label = true;
    let program_base_address = calculate_program_base_address(blob)?;
    println!("Program base address: {:?}", program_base_address);
    for (nth_instruction, instruction) in instructions.iter().enumerate() {
        let instruction_offset = byte_offsets[nth_instruction];
        let offset = instruction_offset + program_base_address;
        let (hex_representation, _) = serialize_instruction(&instruction);

        let instruction_s = if instruction.opcode() == polkavm_common::program::Opcode::fallthrough {
            format_jump_target(jump_target_counter + 1)
        } else {
            instruction.to_string()
        };
        let opcode_name = format!("{:?}", instruction.opcode());

        let line_program = match blob.get_debug_line_program_at(nth_instruction as u32) {
            Ok(line_program) => line_program,
            Err(error) => {
                bail!("failed to parse line program: {error}");
            }
        };

        if let Some(mut line_program) = line_program {
            if last_line_program_entry != Some(line_program.entry_index()) {
                if nth_instruction != 0 {
                    if let Err(error) = writeln!(&mut writer) {
                        bail!("failed to write to output: {error}");
                    }
                }

                last_line_program_entry = Some(line_program.entry_index());
                loop {
                    let region = match line_program.run() {
                        Ok(Some(region)) => region,
                        Ok(None) => break,
                        Err(error) => {
                            bail!("failed to parse line program: {error}");
                        }
                    };

                    if region.instruction_range().contains(&(nth_instruction as u32)) {
                        let frame = region.frames().next().unwrap();
                        let full_name = match frame.full_name() {
                            Ok(full_name) => full_name,
                            Err(error) => {
                                bail!("failed to parse line program: {error}");
                            }
                        }
                        .to_string();

                        if last_full_name != full_name {
                            if let Err(error) = writeln!(&mut writer, "<{}>:", full_name) {
                                bail!("failed to write to output: {error}");
                            }

                            last_full_name = full_name;
                        }

                        break;
                    }
                }
            }
        } else {
            if !last_full_name.is_empty() {
                if let Err(error) = writeln!(&mut writer) {
                    bail!("failed to write to output: {error}");
                }
            }

            last_line_program_entry = None;
            last_full_name.clear();
        }

        if pending_label {
            pending_label = false;
            let result = if !matches!(format, DisassemblyFormat::DiffFriendly) {
                writeln!(&mut writer, "      : {}", format_jump_target(jump_target_counter))
            } else {
                writeln!(&mut writer, "    {}", format_jump_target(jump_target_counter))
            };

            if let Err(error) = result {
                bail!("failed to write to output: {error}");
            }
        }

        if matches!(format, DisassemblyFormat::DiffFriendly) {
            let mut string = instruction_s;
            if let polkavm_common::program::Instruction::load_imm(dst, _) = instruction {
                string = format!("{} = _", dst);
            }

            if let Some(index) = string.find('@') {
                let length = string[index + 1..]
                    .chars()
                    .take_while(|character| character.is_ascii_digit() || matches!(character, 'a' | 'b' | 'c' | 'd' | 'e' | 'f'))
                    .count();
                string.replace_range(index + 1..index + 1 + length, "_");
            }

            if let Some(index_1) = string.find("[0x") {
                let index_2 = string[index_1..].find(']').unwrap() + index_1;
                string.replace_range(index_1..index_2 + 1, "[_]");
            }

            if let Err(error) = writeln!(&mut writer, "    {}", string) {
                bail!("failed to write to output: {error}");
            }
        } else if matches!(format, DisassemblyFormat::Guest | DisassemblyFormat::GuestAndNative) {
            if let Err(error) = writeln!(
                &mut writer,
                "{:08x}: {:<31} {:<31} {}",
                offset,
                hex_representation,
                instruction_s,
                opcode_name
            ) {
                bail!("failed to write to output: {error}");
            }
        }

        if matches!(format, DisassemblyFormat::Native | DisassemblyFormat::GuestAndNative) {
            let (code_origin, code, map) = native.as_ref().unwrap();
            let code_position = map[nth_instruction] as usize;
            let next_code_position = map[nth_instruction + 1] as usize;
            let length = next_code_position - code_position;
            if length != 0 {
                let code_chunk = &code[code_position..next_code_position];
                if let Err(error) = fmt.emit(
                    matches!(format, DisassemblyFormat::GuestAndNative),
                    *code_origin,
                    code_chunk,
                    code_position,
                    &mut writer,
                ) {
                    bail!("failed to write to output: {error}");
                }
            }
        }

        if instruction.opcode().starts_new_basic_block() {
            if instruction.opcode() != polkavm_common::program::Opcode::fallthrough && nth_instruction + 1 != instructions.len() {
                pending_label = true;
            }
            jump_target_counter += 1;
        }
    }

    if let Err(error) = writer.flush() {
        bail!("failed to write to output: {error}");
    }

    Ok(())
}
