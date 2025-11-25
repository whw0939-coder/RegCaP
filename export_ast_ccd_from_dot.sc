import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.util.Using
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption.{APPEND, CREATE}
import scala.util.matching.Regex

def prettyPrintJson(jsonStr: String): String = {
  val sb = new StringBuilder
  var indent = 0
  var inString = false
  var escape = false
  jsonStr.foreach { c =>
    if (escape) { sb.append(c); escape = false }
    else c match {
      case '\\' => sb.append(c); escape = true
      case '"'  => sb.append(c); inString = !inString
      case '{' | '[' =>
        sb.append(c); if (!inString) { sb.append("\n"); indent += 2; sb.append(" " * indent) }
      case '}' | ']' =>
        if (!inString) { sb.append("\n"); indent -= 2; sb.append(" " * indent) }
        sb.append(c)
      case ',' =>
        sb.append(c); if (!inString) { sb.append("\n"); sb.append(" " * indent) }
      case ':' =>
        sb.append(c); if (!inString) sb.append(" ")
      case other => sb.append(other)
    }
  }
  sb.toString
}

def log_error(file_path: String, err_message: String): Unit = {
  val split_path = file_path.split('/')
  val out_path_base = split_path.slice(0, split_path.length - 3).mkString("/")
  val errPath = Paths.get(out_path_base, "log", "graph_export.log")
  try {
    Files.createDirectories(errPath.getParent)
    val line = s"$file_path\n    $err_message\n"
    Files.write(errPath, line.getBytes(StandardCharsets.UTF_8), CREATE, APPEND)
  } catch { case _: Throwable => () }
}

def listCFiles(dir: File): Seq[File] =
  Option(dir.listFiles).toSeq.flatten.filter(f => f.isFile && f.getName.toLowerCase.endsWith(".c"))

def listImmediateSubdirs(dir: File): Seq[File] =
  Option(dir.listFiles).toSeq.flatten.filter(_.isDirectory)

val EdgeRe: Regex = "\"(\\d+)\"\\s*->\\s*\"(\\d+)\"".r
def parseDotEdges(dotText: String): List[(Long, Long)] =
  EdgeRe.findAllMatchIn(dotText).map(m => (m.group(1).toLong, m.group(2).toLong)).toList

def pickDotFile(outDir: File, kind: String, defaultIndex: Int): Option[File] = {
  val files = Option(outDir.listFiles).toSeq.flatten
    .filter(f => f.isFile && f.getName.matches(s"\\d+-$kind\\.dot"))
    .sortBy(f => f.getName.split("-")(0).toInt)
  if (files.isEmpty) None
  else if (defaultIndex < files.length) Some(files(defaultIndex))
  else Some(files.last)
}

def export_wrapper(file_path: String, output_dir: String): Unit = {
  val file_name = new File(file_path).getName
  val out_file_name =
    if (file_name.toLowerCase.endsWith(".c")) file_name.dropRight(2) + ".json"
    else file_name + ".json"
  val outFile = new File(output_dir, out_file_name)

  if (outFile.exists() && outFile.length() > 0) {
    println(s"[skip] ${file_path} -> ${outFile.getAbsolutePath}")
  } else {
    try export_one(file_path, output_dir)
    catch { case e: Throwable => log_error(file_path, "joern error:\n        " + e) }
  }
}

def processBase(srcBase: String, outBase: String): Unit = {
  val srcRoot  = new File(srcBase)
  val outRoot  = new File(outBase)
  outRoot.mkdirs()

  val subdirs = listImmediateSubdirs(srcRoot)
  println(s"[processBase] src=$srcBase  out=$outBase  subdirs=${subdirs.map(_.getName).mkString(",")}")

  subdirs.foreach { sub =>
    val subName = sub.getName
    val outDir  = new File(outRoot, subName)
    outDir.mkdirs()

    val cFiles = listCFiles(sub)
    println(s"  - ${subName}: ${cFiles.size} .c files")
    cFiles.foreach { f => export_wrapper(f.getAbsolutePath, outDir.getAbsolutePath) }
  }
}

@main def export_ast_ccd_from_dot(
  srcBaseArg: String = "",
  outBaseArg: String = ""
): Unit = {
  val defaultSrc = "./Location/"
  val defaultOut = "./Location_js/"

  val srcBase = Option(srcBaseArg).filter(_.nonEmpty)
    .orElse(sys.env.get("SRC_BASE"))
    .getOrElse(defaultSrc)

  val outBase = Option(outBaseArg).filter(_.nonEmpty)
    .orElse(sys.env.get("OUT_BASE"))
    .getOrElse(defaultOut)

  processBase(srcBase, outBase)
  println(s"[done] Exported to mirror under: $outBase")
}
