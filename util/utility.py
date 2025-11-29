import os
import re
import zipfile
import tempfile
from io import BytesIO
from typing import List, Dict
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

import base64

# --- 1. Java Code Processing Functions (Unchanged) ---
def parse_java_zip(uploaded_file: BytesIO) -> Dict[str, str]:
    # ... (function body remains the same)
    java_files = {}
    with zipfile.ZipFile(uploaded_file, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.java') and not name.startswith('__MACOSX/'):
                with zf.open(name) as file:
                    try:
                        content = file.read().decode('utf-8')
                        java_files[name] = content
                    except UnicodeDecodeError:
                        print(f"Skipping non-UTF-8 file: {name}")
    return java_files


# -----------------------------
# Helper: structural split (HYBRID)
# -----------------------------

def split_java_by_structure(text: str) -> List[str]:
    """
    Hybrid splitter (class-level -> method-level -> recursive character splitting)

    - Detects top-level / inner / static classes and enums using an improved regex
    - Attempts to slice each class into method-level chunks where possible
    - Falls back to a RecursiveCharacterTextSplitter for large blocks
    - Logs reasons why certain blocks couldn't be split (anonymous classes, unparsable areas)

    Returns a list of blocks (each block is suitable for downstream chunking/embedding).
    """
    import logging
    # configure lightweight logging for debugging split issues
    logger = logging.getLogger("java_splitter")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[java_splitter] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # 1. capture file-level header (package + imports + file comments)
    header_match = re.match(r"^\s*(?:/\*.*?\*/\s*)*(?:package[\s\S]*?;)?(?:\s*import[\s\S]*?;)*", text, flags=re.DOTALL)
    header = header_match.group(0).strip() if header_match else ""

    # 2. improved regex to detect class/interface/enum declarations (including static/inner)
    decl_pattern = re.compile(r"(?=^[ \t]*(?:public|protected|private|static|final|abstract)?[ \t]*(?:class|interface|enum)\s+[A-Za-z_][A-Za-z0-9_<>]*)", re.MULTILINE)

    matches = list(decl_pattern.finditer(text))
    if not matches:
        logger.info("No class/interface/enum declarations found; returning entire file as single block.")
        return [text]

    indices = [m.start() for m in matches]
    indices.append(len(text))

    blocks: List[str] = []

    # local recursive splitter (used as fallback)
    try:
        recursive_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JAVA, chunk_size=1500, chunk_overlap=150)
    except Exception:
        recursive_splitter = None
        logger.warning("LangChain RecursiveCharacterTextSplitter not available; method-level splitting will be used without fallback character-splitting.")

    # helper: detect anonymous class presence
    anon_pattern = re.compile(r"new\s+[A-Za-z_][A-Za-z0-9_<>]*\s*\([^\)]*\)\s*\{", re.MULTILINE)

    # helper: method-level splitter (conservative regex)
    method_pattern = re.compile(
        r"(?=^[ \t]*(?:public|protected|private|static|final|synchronized|native|abstract)?[ \t]*(?:<[^>]+>\s*)?(?:[A-Za-z_][A-Za-z0-9_<>&\[\] ,.?]+)\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)\s*(?:throws[^{]+)?\{)",
        re.MULTILINE
    )

    for i in range(len(indices) - 1):
        s = indices[i]
        e = indices[i + 1]
        block = text[s:e].strip()
        if header and header not in block:
            block = header + "\n\n" + block

        # quick sanity check: ensure braces roughly balance for this block
        open_braces = block.count("{")
        close_braces = block.count("}")
        if open_braces != 0 and open_braces == close_braces:
            brace_balanced = True
        else:
            brace_balanced = False
        if not brace_balanced:
            logger.warning(f"Brace imbalance detected in block starting at {s} (open={open_braces}, close={close_braces}). Will attempt method-level split or fallback.)")

        # If anonymous classes exist inside, log it (they're harder to split reliably)
        if anon_pattern.search(block):
            logger.info(f"Anonymous/inline class detected inside block starting at {s}. This may affect documentation granularity.")

        # Attempt method-level splitting
        method_matches = list(method_pattern.finditer(block))
        if method_matches:
            method_indices = [m.start() for m in method_matches]
            method_indices.append(len(block))
            # If the first method does not start at 0, include the header portion as a chunk
            if method_indices[0] > 0:
                header_chunk = block[:method_indices[0]].strip()
                if header_chunk:
                    if recursive_splitter:
                        blocks.extend(recursive_splitter.split_text(header_chunk))
                    else:
                        blocks.append(header_chunk)

            # slice each method/body region
            for mi in range(len(method_indices) - 1):
                ms = method_indices[mi]
                me = method_indices[mi + 1]
                method_chunk = block[ms:me].strip()
                # small safety: if method_chunk huge, further recursive split it
                if recursive_splitter and len(method_chunk) > 2000:
                    blocks.extend(recursive_splitter.split_text(method_chunk))
                else:
                    blocks.append(method_chunk)
        else:
            # no method-level matches -> fall back to recursive character-based splitting for the whole block
            if recursive_splitter:
                subchunks = recursive_splitter.split_text(block)
                blocks.extend(subchunks)
                logger.info(f"Block starting at {s} split into {len(subchunks)} recursive chunks.")
            else:
                blocks.append(block)
                logger.info(f"Block starting at {s} added as single chunk (no recursive splitter).")

    # post-process: attempt to merge tiny chunks with neighbours to avoid useless tiny fragments
    merged: List[str] = []
    MIN_LEN = 200  # characters
    for ch in blocks:
        if merged and len(ch) < MIN_LEN:
            merged[-1] = merged[-1] + "\n\n" + ch
        else:
            merged.append(ch)

    logger.info(f"Hybrid splitting produced {len(merged)} blocks from source (header present={bool(header)}).")
    return merged


# In app.py, modify the parse_and_save_springboot_code function:

def parse_and_save_springboot_code(generated_files: dict, generated_test_files: dict) -> str:
    """
    Saves the dictionary of generated files to a temporary directory.
    Includes ULTRA-aggressive cleanup to remove all file tree and non-standard characters from paths.
    """
    if not generated_files:
        return "Generation Failed."

    temp_dir = tempfile.mkdtemp(prefix="springboot_project_")

    # NEW ULTRA-AGGRESSIVE REGEX: Targets vertical lines, corners, dashes, and other unicode clutter.
    # Includes common suspects: │, ├, ─, └, and non-breaking spaces
    #TREE_CHARS_PATTERN = re.compile(r'^[│\s]*[├───└\-\–\—\s\t\ufeff]+', re.UNICODE)

    TREE_CHARS_PATTERN = re.compile(
        r'^[│\s]*[├───└\-\–\—\s\t\n\r\xa0\ufeff\u2000-\u200A\u202F\u205F\u3000]+',
        re.UNICODE
    )

    base64_successful = False
    # Define the Base64 markers
    B64_START = "---BASE64_START---"
    B64_END = "---BASE64_END---"

    # Define the pattern to catch the instruction preamble when the model fails to encode
    # Targets the system prompt's instructions that might precede the code.
    # We look for large chunks of text that shouldn't be code.
    PREAMBLE_PATTERN = re.compile(
        r'(.*?)?(\s*You are an expert Spring Boot developer.*'  # Matches system prompt
        r'|Generate the full source code for.*'  # Matches user prompt
        r'|.*encoding\. Your final response MUST ONLY contain.*'  # Matches encoding instruction
        r')',
        re.DOTALL | re.IGNORECASE
    )

    for relative_path, file_content in generated_files.items():
        try:

            # 1. Initial cleanup and aggressive stripping
            cleaned_path = relative_path.strip()

            # Apply regex to remove tree drawing characters at the start of the path
            cleaned_path = TREE_CHARS_PATTERN.sub('', cleaned_path)

            # Final trim to ensure no leading/trailing spaces remain
            cleaned_path = cleaned_path.strip()

            # CRITICAL CHECK: Ensure the path is still valid
            if not cleaned_path or cleaned_path.startswith('/'):
                st.warning(
                    f"Skipping path '{relative_path}' after cleanup, resulting in invalid path: '{cleaned_path}'")
                continue

            # 2. Use the cleaned path for saving
            print(cleaned_path)
            pattern = re.compile(r'[│├└─]+\s*')
            cleaned_path  = pattern.sub('', cleaned_path)
            print("Cleaned path after applying pattern")
            print(cleaned_path)

            full_path = os.path.join(temp_dir, cleaned_path)

            # --- Code De-obfuscation / Reverse Sanitization (Existing Logic) ---
            # ... (Existing de-obfuscation logic using the cleaned_path remains here) ...
            if cleaned_path.endswith('.java'):
                # Revert the custom prefixes for REST annotations
                # Example: @@RestController -> @RestController
                file_content = file_content.replace("@@Rest", "@Rest")
                file_content = file_content.replace("@@Request", "@Request")
                file_content = file_content.replace("@@Autowired", "@Autowired")
                file_content = file_content.replace("@@Service", "@Service")
                file_content = file_content.replace("@@Repository", "@Repository")
                file_content = file_content.replace("@@Auto", "@Auto")
                file_content = file_content.replace("@@Arepository", "@Repository")
                file_content = file_content.replace("@@Service", "@Service")
                file_content = file_content.replace("@@Kervice", "@Service")
                file_content = file_content.replace("@public", "public")



                # Add any other prefixes you used, like @@Service, @@Component, etc.

                # B. Reverse POM.xml Tags
            if cleaned_path == 'pom.xml':
                # Revert the custom dependency tags
                # Example: <dep_tag> -> <dependency>
                file_content = file_content.replace("<dep_tag>", "<dependency>")
                file_content = file_content.replace("</dep_tag>", "</dependency>")
                # Add any other XML tags you obfuscated

            # Remove markdown block syntax
            file_content = re.sub(r'^\s*```[a-z]*\s*\n|\s*```\s*$', '', file_content.strip(), flags=re.MULTILINE)

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write the content to the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

        except Exception as e:
            st.error(f"Error saving file {relative_path} (cleaned to {cleaned_path}): {e}")
            continue

    for relative_path, file_content in generated_test_files.items():
        try:

            # 1. Initial cleanup and aggressive stripping
            cleaned_path = relative_path.strip()

            # Apply regex to remove tree drawing characters at the start of the path
            cleaned_path = TREE_CHARS_PATTERN.sub('', cleaned_path)

            # Final trim to ensure no leading/trailing spaces remain
            cleaned_path = cleaned_path.strip()

            # CRITICAL CHECK: Ensure the path is still valid
            if not cleaned_path or cleaned_path.startswith('/'):
                st.warning(
                    f"Skipping path '{relative_path}' after cleanup, resulting in invalid path: '{cleaned_path}'")
                continue

            # 2. Use the cleaned path for saving
            print(cleaned_path)
            pattern = re.compile(r'[│├└─]+\s*')
            cleaned_path  = pattern.sub('', cleaned_path)
            print("Cleaned path after applying pattern")
            print(cleaned_path)

            # --- FIX FOR TEST FILE TARGET DIRECTORY ---
            # Force all test files to go to src/test/java/
            #test_base = os.path.join(temp_dir, "src", "test", "java")
            # Strip any accidental existing test path
            cleaned_rel = cleaned_path.replace("src/main/java/", "src/test/java/").lstrip("/")

            full_path = os.path.join(temp_dir, cleaned_rel)
            #full_path = os.path.join(temp_dir, cleaned_path)

            # --- Code De-obfuscation / Reverse Sanitization (Existing Logic) ---
            # ... (Existing de-obfuscation logic using the cleaned_path remains here) ...
            if cleaned_path.endswith('.java'):
                # Revert the custom prefixes for REST annotations
                # Example: @@RestController -> @RestController
                file_content = file_content.replace("@@Rest", "@Rest")
                file_content = file_content.replace("@@Request", "@Request")
                file_content = file_content.replace("@@Autowired", "@Autowired")
                file_content = file_content.replace("@@Service", "@Service")
                file_content = file_content.replace("@@Repository", "@Repository")
                file_content = file_content.replace("@@Auto", "@Auto")
                file_content = file_content.replace("@@Arepository", "@Repository")
                file_content = file_content.replace("@@Service", "@Service")
                file_content = file_content.replace("@@Kervice", "@Service")
                file_content = file_content.replace("@public", "public")




            # Remove markdown block syntax
            file_content = re.sub(r'^\s*```[a-z]*\s*\n|\s*```\s*$', '', file_content.strip(), flags=re.MULTILINE)

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write the content to the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

        except Exception as e:
            st.error(f"Error saving file {relative_path} (cleaned to {cleaned_path}): {e}")
            continue

    return temp_dir

def extract_file_list_from_blueprint(blueprint_text: str) -> list:
    file_paths = []
    VALID_EXTENSIONS = ('.java', '.xml', '.properties', '.yml')
    BASE_PACKAGE_PATH = 'com/codeanalyzer/'  # Use path structure for easy concatenation

    for line in blueprint_text.split('\n'):
        line = line.strip()

        # Check if the line ends with a known file extension
        is_valid_file_line = any(line.endswith(ext) for ext in VALID_EXTENSIONS)

        if is_valid_file_line or line.startswith('pom.xml'):

            # Remove any numbering, bullets, or leading text (more aggressive cleanup)
            path = re.sub(r'^\s*[\d\.\*\-\–\—\s]+\s*', '', line).strip('`"., ')

            # --- START: Mandatory Structure Reconstruction ---

            if path and path.endswith('.java'):
                # 1. Strip off any extraneous package prefix the LLM might have included
                path = path.replace('src/main/java/', '').replace(BASE_PACKAGE_PATH, '').strip('/')

                # 2. Heuristic to rebuild the full path based on component type
                package_folder = ''

                # Use simple case-insensitive checks
                if 'Controller' in path or path.endswith('Controller.java'):
                    package_folder = 'controller/'
                elif 'Service' in path or path.endswith('Service.java'):
                    package_folder = 'service/'
                elif 'Repository' in path or path.endswith('Repository.java'):
                    package_folder = 'repository/'
                elif 'Model' in path or 'Entity' in path:
                    package_folder = 'model/'
                elif 'config' in path or 'Config' in path:
                    package_folder = 'config/'
                elif 'dto' in path or 'Dto' in path or path.endswith('DTO.java'):
                    package_folder = 'dto/'
                elif 'util' in path or 'Util' in path:
                    package_folder = 'util/'
                elif 'entity' in path:
                    package_folder = 'model/'
                elif 'exception' in path or 'Exception' in path or path.endswith('Exception.java'):
                    package_folder = 'exception/'

                # Default/Application class
                elif path.endswith('Application.java'):
                    package_folder = ''  # Goes directly under base package

                # 3. Force the complete, mandatory Spring Boot file structure
                full_java_path = f'src/main/java/{BASE_PACKAGE_PATH}{package_folder}{path}'
                file_paths.append(full_java_path.replace('//', '/'))

            elif path and (path.endswith('.xml') or path.endswith('.properties') or path.endswith('.yml')):
                # Handle config/resource files
                if path == 'pom.xml':
                    file_paths.append('pom.xml')
                else:
                    # Force resource path
                    resource_path = f'src/main/resources/{path}'
                    file_paths.append(resource_path.replace('//', '/'))

    # Final cleanup and ensuring mandatory files are present
    final_paths = set(file_paths)

    if not any(f.endswith('pom.xml') for f in final_paths):
        final_paths.add('pom.xml')
    if not any(('application.properties' in f or 'application.yml' in f) for f in final_paths):
        final_paths.add('src/main/resources/application.properties')

    return list(final_paths)


