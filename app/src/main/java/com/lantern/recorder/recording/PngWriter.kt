package com.lantern.recorder.recording

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.util.zip.CRC32
import java.util.zip.Deflater

/**
 * Minimal grayscale PNG encoder. Exists because [android.graphics.Bitmap] cannot
 * compress to 16-bit grayscale — and the raw depth map must be stored losslessly
 * as 16-bit (millimeters) for the offline reconstruction pipeline.
 *
 * Supports 8-bit and 16-bit single-channel (grayscale, PNG color type 0) images.
 * Sample bytes must be supplied big-endian and row-major, as PNG requires.
 */
object PngWriter {

    private val SIGNATURE = byteArrayOf(
        137.toByte(), 80, 78, 71, 13, 10, 26, 10,
    )

    /**
     * @param samplesBigEndian width*height*(bitDepth/8) bytes, row-major, big-endian.
     * @param bitDepth 8 or 16.
     */
    fun writeGrayscale(file: File, samplesBigEndian: ByteArray, width: Int, height: Int, bitDepth: Int) {
        require(bitDepth == 8 || bitDepth == 16) { "bitDepth must be 8 or 16" }
        val bytesPerSample = bitDepth / 8
        val rowBytes = width * bytesPerSample
        require(samplesBigEndian.size >= rowBytes * height) { "sample buffer too small" }

        // Prepend a per-scanline filter byte (0 = None).
        val raw = ByteArray((rowBytes + 1) * height)
        var src = 0
        var dst = 0
        for (y in 0 until height) {
            raw[dst++] = 0
            System.arraycopy(samplesBigEndian, src, raw, dst, rowBytes)
            src += rowBytes
            dst += rowBytes
        }

        val idat = deflate(raw)

        BufferedOutputStream(FileOutputStream(file)).use { out ->
            out.write(SIGNATURE)

            val ihdr = ByteArray(13)
            writeIntBE(ihdr, 0, width)
            writeIntBE(ihdr, 4, height)
            ihdr[8] = bitDepth.toByte()
            ihdr[9] = 0 // color type 0 = grayscale
            ihdr[10] = 0 // compression
            ihdr[11] = 0 // filter
            ihdr[12] = 0 // interlace
            writeChunk(out, "IHDR", ihdr)
            writeChunk(out, "IDAT", idat)
            writeChunk(out, "IEND", ByteArray(0))
            out.flush()
        }
    }

    private fun deflate(data: ByteArray): ByteArray {
        val deflater = Deflater(Deflater.BEST_SPEED)
        deflater.setInput(data)
        deflater.finish()
        val buffer = ByteArray(64 * 1024)
        val builder = ArrayList<ByteArray>()
        var total = 0
        while (!deflater.finished()) {
            val n = deflater.deflate(buffer)
            if (n > 0) {
                builder.add(buffer.copyOf(n))
                total += n
            }
        }
        deflater.end()
        val out = ByteArray(total)
        var pos = 0
        for (chunk in builder) {
            System.arraycopy(chunk, 0, out, pos, chunk.size)
            pos += chunk.size
        }
        return out
    }

    private fun writeChunk(out: OutputStream, type: String, data: ByteArray) {
        val typeBytes = type.toByteArray(Charsets.US_ASCII)
        val length = ByteArray(4)
        writeIntBE(length, 0, data.size)
        out.write(length)
        out.write(typeBytes)
        out.write(data)

        val crc = CRC32()
        crc.update(typeBytes)
        crc.update(data)
        val crcBytes = ByteArray(4)
        writeIntBE(crcBytes, 0, crc.value.toInt())
        out.write(crcBytes)
    }

    private fun writeIntBE(dest: ByteArray, offset: Int, value: Int) {
        dest[offset] = (value ushr 24 and 0xff).toByte()
        dest[offset + 1] = (value ushr 16 and 0xff).toByte()
        dest[offset + 2] = (value ushr 8 and 0xff).toByte()
        dest[offset + 3] = (value and 0xff).toByte()
    }
}
